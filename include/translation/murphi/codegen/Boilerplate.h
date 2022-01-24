#pragma once
#include "inja/inja.hpp"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/raw_ostream.h"

namespace murphi {
namespace boilerplate {
void setBoilerplateConstants(nlohmann::json &data);
void setBoilerplateAccessType(nlohmann::json &data);

nlohmann::json getMessageHandlerSwitch();
nlohmann::json getStateHandlerSwitch() ;

} // namespace boilerplate
} // namespace murphi
