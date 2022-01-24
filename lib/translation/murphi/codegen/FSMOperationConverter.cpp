#include "translation/murphi/codegen/FSMOperationConverter.h"
#include "translation/murphi/codegen/MurphiCodeGen.h"

using namespace nlohmann;
using namespace mlir;
using namespace mlir::fsm;
using namespace murphi;

namespace murphi {

nlohmann::json FSMOperationConverter::convert(mlir::fsm::TransitionOp op) {
  setupSymbolTable(op);
  json data = json::array();
  for (auto &nestedOp : op.getBody().getOps()) {
    data.push_back(convert(&nestedOp));
  }
  return data;
}

nlohmann::json FSMOperationConverter::convert(mlir::Operation *op) {
  if (auto msgOp = dyn_cast<MessageOp>(op))
    return convert(msgOp);

  assert(0 && "Trying to convert an invalid operation");
}

void FSMOperationConverter::setupSymbolTable(mlir::fsm::TransitionOp op) {
  auto parentMach = op->getParentOfType<MachineOp>();
  // for each state variable
  for (auto stateVar : parentMach.getOps<VariableOp>()) {
    auto varName = stateVar.name();
    std::string mappedMurphiValue =
        std::string{murphi::detail::cle_a} + "." + varName.str();
    symbolTable.insert(stateVar, mappedMurphiValue);
  }

  // if the transition op has an input message argument
  if (op.getArguments().size() == 1)
    symbolTable.insert(op.getArgument(0), murphi::detail::c_inmsg);
}

json FSMOperationConverter::convert(MessageOp op) {
  // default :: i.e. Resp(GetM)
  auto msgConstr =
      murphi::detail::ProcCall{op.msgType().getLeafReference().str(),
                               {murphi::detail::ExprID{op.msgName().str()}}};

  // now we add the remainder of the parameters
  for (auto operand : op->getOperands()) {
    auto val = symbolTable.lookup(operand);
    msgConstr.actuals.push_back(murphi::detail::ExprID{std::move(val)});
  }

  // create the assignment to msg;
  auto msgAssign = murphi::detail::Assignment<murphi::detail::Designator,
                                              decltype(msgConstr)>{
      {murphi::detail::c_msg, {}}, std::move(msgConstr)};

  return msgAssign;
}

} // namespace murphi