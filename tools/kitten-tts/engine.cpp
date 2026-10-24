#include "engine.h"
#include "llama.h"
#include "text.h"
#include "sampling.h"
#include "decoder.h"
#include "assets.h"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <stdexcept>
namespace kitten {
using clock_type = std::chrono::steady_clock;
static json read_json(const std::string & path) {
    std::ifstream f(path); if(!f) throw std::runtime_error("cannot open "+path); json j; f>>j;return j;
}
static std::vector<int> tokenize(const llama_vocab * vocab,const std::string & text) {
    int n=llama_tokenize(vocab,text.data(),text.size(),nullptr,0,false,true);
    if(n==0) return {};
    std::vector<int> ids(-n); n=llama_tokenize(vocab,text.data(),text.size(),ids.data(),ids.size(),false,true);
    if(n<0) throw std::runtime_error("tokenization failed");
    ids.resize(n);return ids;
}
struct batch {
    llama_batch b;
    batch(int n,int dim=0): b(llama_batch_init(n,dim,1)) {}
    ~batch(){llama_batch_free(b);}
    void position(int i,int pos,bool logits) {b.pos[i]=pos;b.n_seq_id[i]=1;b.seq_id[i][0]=0;b.logits[i]=logits;}
};
static void check_known(const args_map & args,const std::vector<std::string> & known) {
    for(auto & a:args) if(std::find(known.begin(),known.end(),a.first)==known.end()) throw std::invalid_argument("unknown option "+a.first);
}
static assets resolve(const args_map & args) {
    check_known(args,{"--threads","--decoder-threads","--data","--ca-file","--assets","--repo","--revision","--cache-dir","--decoder","--hf-token","--model","--offline","--no-repack"});
    // the HTTP client loads system CA paths, and OpenSSL reads SSL_CERT_FILE there
    if(args.count("--ca-file")) setenv("SSL_CERT_FILE",args.at("--ca-file").c_str(),1);
    return resolve_assets(args);
}
struct engine::impl {
    kitten_text_processing::Normalizer normalizer;
    assets files;
    json voices;
    token_map tm;
    std::unique_ptr<llama_model,decltype(&llama_model_free)> model{nullptr,llama_model_free};
    std::unique_ptr<decoder> dec;
    int threads=8;
    explicit impl(const args_map & args): normalizer(args.at("--data")), files(resolve(args)), voices(read_json(files.voices)), tm(files.config.at("token_map")) {}
};
engine::engine(const args_map & args) {
    if(!args.count("--data")) throw std::invalid_argument("--data is required");
    p=std::make_unique<impl>(args);
    auto get=[&](const std::string & key,const std::string & fallback){auto it=args.find(key);return it==args.end()?fallback:it->second;};
    p->threads=std::stoi(get("--threads","8"));int decoder_threads=std::stoi(get("--decoder-threads",std::to_string(p->threads)));
    if(p->threads<1 || decoder_threads<1) throw std::invalid_argument("thread counts must be positive");
    static std::once_flag backend_once;
    std::call_once(backend_once,[]{ggml_backend_load_all();llama_backend_init();});
    auto mp=llama_model_default_params();mp.n_gpu_layers=0;mp.use_extra_bufts=!args.count("--no-repack");
    p->model.reset(llama_model_load_from_file(p->files.model.c_str(),mp));
    if(!p->model) throw std::runtime_error("cannot load language model");
    if(p->tm.base+p->tm.count>llama_vocab_n_tokens(llama_model_get_vocab(p->model.get()))) throw std::runtime_error("token map exceeds vocabulary");
    p->dec=std::make_unique<decoder>(p->files.decoder,decoder_threads,0);
}
engine::~engine() = default;
std::string engine::metadata() const {
    json names=json::array();
    for(auto & v:p->voices.items()) names.push_back(v.key());
    return json{{"config",p->files.config},{"voices",names}}.dump();
}
std::string engine::normalize(const std::string & text) const { return kitten::normalize(text,p->normalizer,true); }
std::vector<float> engine::generate(const args_map & args,std::string & report_json,const chunk_callback & on_chunk) {
    check_known(args,{"--text","--voice","--preset","--max-tokens","--no-normalize","--use-reference","--use-emotion","--seed","--temperature","--top-k","--top-p","--min-p","--repetition-penalty","--repetition-window","--run-penalty","--run-grace","--chunk-chars","--chunk-min-chars","--chunk-gap"});
    if(!args.count("--text")) throw std::invalid_argument("--text is required");
    auto get=[&](const std::string & key,const std::string & fallback){auto it=args.find(key);return it==args.end()?fallback:it->second;};
    const auto & config=p->files.config; const auto & tm=p->tm;
    std::string voice=get("--voice",config.value("default_voice","Bruno"));
    if(!p->voices.contains(voice)) throw std::invalid_argument("unknown voice: "+voice);
    const auto & ref=p->voices.at(voice);
    int seed=std::stoi(get("--seed",std::to_string(std::random_device{}() % 2147483647))), max_tokens=std::stoi(get("--max-tokens","1000"));
    if(max_tokens<1) throw std::invalid_argument("token count must be positive");
    auto gen=config.value("generation",json::object());
    int chunk_chars=std::stoi(get("--chunk-chars",std::to_string(gen.value("chunk_chars",380))));
    int chunk_min=std::stoi(get("--chunk-min-chars",std::to_string(gen.value("chunk_min_chars",130))));
    float gap=std::stof(get("--chunk-gap","0.16"));
    if(!std::isfinite(gap) || gap<0 || gap>60) throw std::invalid_argument("invalid chunk gap");
    auto spoken=kitten::normalize(args.at("--text"),p->normalizer,!args.count("--no-normalize"));
    auto chunks=kitten::split(spoken,chunk_chars,chunk_min);
    if(chunks.empty()) throw std::invalid_argument("no text to speak");
    auto preset=config.at("decode_presets").at(get("--preset",config.value("default_preset","stable")));
    sampling settings;
    settings.temperature=std::stof(get("--temperature",std::to_string(preset.at("temperature").get<float>())));
    settings.top_k=std::stoi(get("--top-k",std::to_string(preset.at("top_k").get<int>())));
    settings.top_p=std::stof(get("--top-p",std::to_string(preset.at("top_p").get<float>())));
    settings.min_p=std::stof(get("--min-p",std::to_string(preset.value("min_p",0.f))));
    settings.repetition=std::stof(get("--repetition-penalty","1.1"));settings.window=std::stoi(get("--repetition-window",std::to_string(gen.value("repetition_window",0))));
    settings.run_penalty=std::stof(get("--run-penalty","1.3"));settings.grace=std::stoi(get("--run-grace","10"));
    if(!std::isfinite(settings.temperature) || settings.temperature<=0 || settings.top_k<0 || !(settings.top_p>0 && settings.top_p<=1) || !(settings.min_p>=0 && settings.min_p<=1) || !std::isfinite(settings.repetition) || settings.repetition<=0 || !std::isfinite(settings.run_penalty) || settings.run_penalty<0 || settings.grace<1) throw std::invalid_argument("invalid sampling settings");
    auto vocab=llama_model_get_vocab(p->model.get());int n_vocab=llama_vocab_n_tokens(vocab);
    auto speaker=ref.at("speaker").get<std::vector<float>>();
    if(speaker.size()!=size_t(llama_model_n_embd(p->model.get()))) throw std::runtime_error("speaker projection dimension mismatch");
    auto ref_text=tokenize(vocab,ref.at("transcript").get<std::string>());
    auto ref_audio=ref.at("reference_tokens").get<std::vector<int>>();
    // unset --use-emotion / --use-reference keep the CLI defaults
    bool use_emotion=args.count("--use-emotion") ? args.at("--use-emotion")!="0" : kitten::has_expression(args.at("--text"));
    bool use_reference=args.count("--use-reference") ? args.at("--use-reference")!="0" : gen.value("use_reference_prompt",true);
    std::vector<int> emotion;
    if(use_emotion && gen.contains("emotion_control")) emotion=tokenize(vocab,gen.at("emotion_control"));
    std::mt19937 rng(seed); p->dec->seed(seed);
    std::vector<std::vector<float>> waves;
    json report={{"normalized",spoken},{"voice",voice},{"sample_rate",24000},{"seed",seed},{"chunks",json::array()}};
    double lm_seconds=0,decoder_seconds=0;size_t total_tokens=0;bool stopped=false;
    for(auto & chunk:chunks) {
        auto ids=kitten::prompt(tm,tokenize(vocab,chunk),ref_text,ref_audio,emotion,use_reference);
        int budget=chunks.size()==1?max_tokens:std::max(200,std::min(max_tokens,int(kitten::characters(chunk)/20.0*25*1.8)));
        auto cp=llama_context_default_params();cp.n_ctx=ids.size()+budget;cp.n_batch=512;cp.n_ubatch=512;cp.n_threads=p->threads;cp.n_threads_batch=p->threads;cp.no_perf=false;
        std::unique_ptr<llama_context,decltype(&llama_free)> ctx(llama_init_from_model(p->model.get(),cp),llama_free);
        if(!ctx) throw std::runtime_error("cannot initialize context");
        auto begin=clock_type::now();
        { batch b(1,speaker.size());b.b.n_tokens=1;std::copy(speaker.begin(),speaker.end(),b.b.embd);b.position(0,0,false);
          if(llama_decode(ctx.get(),b.b)) throw std::runtime_error("speaker prefill failed"); }
        for(size_t offset=1;offset<ids.size();) {
            int n=std::min(size_t(512),ids.size()-offset);batch b(n);b.b.n_tokens=n;
            for(int i=0;i<n;++i) {b.b.token[i]=ids[offset+i];b.position(i,offset+i,offset+i+1==ids.size());}
            if(llama_decode(ctx.get(),b.b)) throw std::runtime_error("prompt prefill failed");
            offset+=n;
        }
        auto history=ids;std::vector<int> audio_ids,generated;bool ended=false;
        for(int step=0;step<budget;++step) {
            const float * raw=llama_get_logits_ith(ctx.get(),-1);
            std::vector<float> scores(raw,raw+n_vocab);kitten::process(scores,history,ids.size(),tm,settings);
            int token=kitten::sample(scores,rng,false);
            generated.push_back(token);history.push_back(token);
            if(token==tm.speech_end || token==tm.stop) {ended=true;break;}
            if(token>=tm.base && token<tm.base+tm.count) audio_ids.push_back(token-tm.base);
            if(step+1<budget) {batch b(1);b.b.n_tokens=1;b.b.token[0]=token;b.position(0,history.size()-1,true);if(llama_decode(ctx.get(),b.b)) throw std::runtime_error("generation failed");}
        }
        double elapsed=std::chrono::duration<double>(clock_type::now()-begin).count();lm_seconds+=elapsed;total_tokens+=generated.size();
        auto perf=llama_perf_context(ctx.get());
        report["chunks"].push_back({{"text",chunk},{"prompt",ids},{"generated",generated},{"audio_tokens",audio_ids},{"terminated",ended},{"lm_seconds",elapsed},
            {"prompt_eval_ms",perf.t_p_eval_ms},{"prompt_eval_tokens",perf.n_p_eval},{"eval_ms",perf.t_eval_ms},{"eval_tokens",perf.n_eval}});
        begin=clock_type::now();waves.push_back(p->dec->decode(audio_ids,ref.dump()));decoder_seconds+=std::chrono::duration<double>(clock_type::now()-begin).count();
        if(on_chunk && !on_chunk(waves.back())) {stopped=true;break;}
    }
    report["lm_seconds"]=lm_seconds;report["decoder_seconds"]=decoder_seconds;report["generated_tokens"]=total_tokens;report["stopped"]=stopped;
    std::vector<float> audio;
    if(!stopped || std::any_of(waves.begin(),waves.end(),[](const auto & w){return !w.empty();})) audio=kitten::join(waves,gap);
    report["audio_seconds"]=audio.size()/24000.0;
    if(!audio.empty()) report["rtf"]=(lm_seconds+decoder_seconds)/(audio.size()/24000.0);
    report_json=report.dump();
    return audio;
}
}
