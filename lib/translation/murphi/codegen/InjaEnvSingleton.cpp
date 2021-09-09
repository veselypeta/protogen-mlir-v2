#include "translation/murphi/codegen/InjaEnvSingleton.h"
#include <inja/inja.hpp>
#include <mutex>

using namespace inja;

inja::Environment *InjaEnvSingleton::env = nullptr;

std::mutex initMut;
void registerInjaCallbacks(Environment &env) {

  /*
   * -- msg_uses_field --
   * Used in the message_constructor template to determine if a message
   * constructor uses the field in the global type
   * i.e. Resp messages use the cl field, but Ack messages do not
   */
  env.add_callback("msg_uses_field", 2, [](Arguments &args) {
    auto tmplData = args.at(0)->get<json>();  // 1st parameter is the msg type
    auto globalMsg = args.at(1)->get<json>(); // global msg type
    auto additionalParameters = tmplData["additionalParameters"];
    return std::find_if(additionalParameters.begin(),
                        additionalParameters.end(), [globalMsg](json elem) {
                          return elem["id"] == globalMsg["id"];
                        }) != additionalParameters.end();
  });

  /*
   * -- render_template --
   * Takes two Arguments
   * (1) a string name for the template held in the templates folder
   * (2) a json object which is passed to the template during rendering
   */
  env.add_callback("render_template", 2, [&env](Arguments &args){
    std::string tmplName = args.at(0)->get<std::string>();
    json data = args.at(1)->get<json>();
    auto tmpl = env.parse_template(tmplName);
    return env.render(tmpl, data);
  });
}

inja::Environment &InjaEnvSingleton::getInstance() {
  std::lock_guard<std::mutex> initLock(initMut);
  if (env == nullptr) {
    env = new inja::Environment{"../../templates/"};
    env->set_trim_blocks(true);
    env->set_lstrip_blocks(true);
    registerInjaCallbacks(*env);
  }
  return *env;
}

InjaEnvSingleton::~InjaEnvSingleton() {
  delete env;
}

