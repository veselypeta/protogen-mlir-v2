#include "translation/murphi/codegen/FSMOperationConverter.h"
#include "translation/murphi/codegen/MurphiCodeGen.h"

using namespace nlohmann;
using namespace mlir;
using namespace mlir::fsm;
using namespace murphi;

namespace murphi {

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

nlohmann::json FSMOperationConverter::convert(mlir::fsm::TransitionOp op) {
  // instantiate a scope
  SymbolTableScopeT scope(symbolTable);
  setupSymbolTable(op);

  json data = json::array();
  for (auto &nestedOp : op.getBody().getOps()) {
    json convertedOp = convert(&nestedOp);
    if (convertedOp == nullptr)
      continue;
    data.push_back(std::move(convertedOp));
  }

  // if has end state attr
  auto nextStateAttr = op.nextState();
  if (nextStateAttr.hasValue()) {
    auto nextState = nextStateAttr.getValue().getLeafReference();
    data.push_back(murphi::detail::Assignment<murphi::detail::Designator,
                                              murphi::detail::ExprID>{
        {murphi::detail::cle_a,
         {murphi::detail::Indexer{
             "object", murphi::detail::ExprID{murphi::detail::c_state}}}},
        {nextState.str()}});
  }

  return data;
}

nlohmann::json FSMOperationConverter::convert(mlir::Operation *op) {
  if (auto msgOp = dyn_cast<MessageOp>(op))
    return convert(msgOp);
  if (auto refOp = dyn_cast<ReferenceOp>(op))
    convert(refOp);
  if (auto accessOp = dyn_cast<AccessOp>(op))
    convert(accessOp);
  if (auto updateOp = dyn_cast<UpdateOp>(op))
    return convert(updateOp);
  return nullptr;
}

json FSMOperationConverter::convert(MessageOp op) {
  // default :: i.e. Resp(adr, GetM)
  auto msgConstr = murphi::detail::ProcCallExpr{
      op.msgType().getLeafReference().str(),
      {murphi::detail::ExprID{murphi::detail::adr_a},
       murphi::detail::ExprID{op.msgName().str()}}};

  // now we add the remainder of the parameters
  for (auto operand : op->getOperands()) {
    auto val = symbolTable.lookup(operand);
    msgConstr.actuals.emplace_back(murphi::detail::ExprID{std::move(val)});
  }

  return murphi::detail::Assignment<murphi::detail::Designator,
                                    decltype(msgConstr)>{
      {murphi::detail::c_msg, {}}, std::move(msgConstr)};
}

nlohmann::json FSMOperationConverter::convert(mlir::fsm::UpdateOp op) {
  auto lhs = symbolTable.lookup(op.variable());
  auto rhs = symbolTable.lookup(op.value());

  return murphi::detail::Assignment<murphi::detail::Designator,
                                    murphi::detail::ExprID>{
      {std::move(lhs), {}}, {std::move(rhs)}};
}

void FSMOperationConverter::convert(mlir::fsm::ReferenceOp op) {
  // if the reference is to the same machine -> resolve to "m"
  // if the reference is to the directory -> resolve to "directory"
  auto reference = op.reference();
  auto refMach =
      op->getParentOfType<ModuleOp>().lookupSymbol<MachineOp>(reference);
  auto parentMach = op->getParentOfType<MachineOp>();

  if (refMach == parentMach) {
    symbolTable.insert(op, murphi::detail::c_mach);
  } else {
    symbolTable.insert(op, murphi::detail::machines.directory.str());
  }
}

void FSMOperationConverter::convert(mlir::fsm::AccessOp op) {
  std::string accessor = symbolTable.lookup(op.msg());
  symbolTable.insert(op, accessor + "." + op.memberId().str());
}

} // namespace murphi