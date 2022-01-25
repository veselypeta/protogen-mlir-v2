#pragma once
#include "inja/inja.hpp"
#include "mlir/IR/BuiltinOps.h"
#include "translation/murphi/codegen/MurphiStructs.h"
#include "llvm/Support/raw_ostream.h"

namespace murphi {
namespace boilerplate {

/*
 * Helper Generating Functions
 */
nlohmann::json emitNetworkDefinitionJson();

void setBoilerplateConstants(nlohmann::json &data);
void setBoilerplateAccessType(nlohmann::json &data);

murphi::detail::SwitchStmt getMessageHandlerSwitch();
murphi::detail::SwitchStmt getStateHandlerSwitch();

murphi::detail::CaseStmt getBasicCaseStmt(const std::string &caseStr);
murphi::detail::CaseStmt getBasicCaseStmt(std::string &&caseText);
} // namespace boilerplate
} // namespace murphi
