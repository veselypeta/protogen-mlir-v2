#include "translation/murphi/codegen/FSMOperationConverter.h"
#include "translation/murphi/codegen/MurphiCodeGen.h"
#include "translation/utils/utils.h"

using namespace nlohmann;
using namespace mlir;
using namespace mlir::fsm;
using namespace murphi;

namespace murphi {

void FSMOperationConverter::setupSymbolTable(mlir::fsm::TransitionOp op) {
  auto parentMach = op->getParentOfType<MachineOp>();
  auto theModule = op->getParentOfType<ModuleOp>();

  // setup global networks
  for (auto network : theModule.getOps<NetworkOp>()) {
    symbolTable.insert(network, network.sym_name().getLeafReference().str());
  }

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
    // remember to mangle the state
    auto nextState = translation::utils::mangleState(
        nextStateAttr.getValue().getLeafReference().str(),
        op->getParentOfType<MachineOp>().sym_name().str() + "_");
    data.push_back(murphi::detail::Assignment<murphi::detail::Designator,
                                              murphi::detail::ExprID>{
        {murphi::detail::cle_a,
         {murphi::detail::Indexer{
             "object", murphi::detail::ExprID{murphi::detail::c_state}}}},
        {nextState}});
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
  if (auto stdConstOp = dyn_cast<ConstantOp>(op))
    convert(stdConstOp);
  if (auto ifOp = dyn_cast<IfOp>(op))
    return convert(ifOp);
  if (auto fsmConst = dyn_cast<ConstOp>(op))
    convert(fsmConst);
  if (auto sendOp = dyn_cast<SendOp>(op))
    return convert(sendOp);
  if (auto compOp = dyn_cast<CompareOp>(op))
    convert(compOp);
  if(auto addOp = dyn_cast<AddOp>(op))
    convert(addOp);
  if(auto setAdd = dyn_cast<SetAdd>(op))
    return convert(setAdd);
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

  // add msg to the symbol table
  symbolTable.insert(op, murphi::detail::c_msg);

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

void FSMOperationConverter::convert(mlir::ConstantOp op) {
  Attribute valAttr = op.value();
  if (auto boolAttr = valAttr.dyn_cast<BoolAttr>())
    symbolTable.insert(op, boolAttr.getValue() ? "true" : "false");
}

void FSMOperationConverter::convert(mlir::fsm::ConstOp op) {
  Attribute valAttr = op.value();
  auto parentMach = op->getParentOfType<MachineOp>();

  if (auto strAttr = valAttr.dyn_cast<StringAttr>()) {
    // we need to remember to mangle the names of the states
    if(op.getResult().getType().isa<StateType>()){
    symbolTable.insert(
        op, translation::utils::mangleState(strAttr.getValue().str(),
                                            parentMach.sym_name().str() + "_"));
    } else{
      symbolTable.insert(
          op, strAttr.getValue().str());
    }
  }
}

void FSMOperationConverter::convert(mlir::fsm::CompareOp op) {
  symbolTable.insert(op, symbolTable.lookup(op.lhs()) + op.compOp().str() +
                             symbolTable.lookup(op.rhs()));
}

nlohmann::json FSMOperationConverter::convert(mlir::fsm::SendOp op) {
  auto netId = symbolTable.lookup(op.network());
  auto sendCall = murphi::detail::ProcCall{
      murphi::detail::send_pref_f + netId,
      {detail::ExprID{symbolTable.lookup(op.message())}}};
  return sendCall;
}

nlohmann::json FSMOperationConverter::convert(mlir::fsm::IfOp op) {
  auto condText = symbolTable.lookup(op.condition());
  auto ifStmt =
      murphi::detail::IfStmt<murphi::detail::ExprID>{{condText}, {}, {}};
  {
    // instantiate a new symbol table -> for the then scope
    SymbolTableScopeT thenScope(symbolTable);
    for (auto &thenOp : op.thenRegion().getOps()) {
      auto convertedStmt = convert(&thenOp);
      if (convertedStmt != nullptr)
        ifStmt.thenStmts.emplace_back(convertedStmt);
    }
  } // then scope deallocated here
  if (!op.elseRegion().empty()) {
    // ST for the else scope
    SymbolTableScopeT elseScope(symbolTable);
    for (auto &elseOp : op.elseRegion().getOps()) {
      auto convertedStmt = convert(&elseOp);
      if (convertedStmt != nullptr)
        ifStmt.elseStmts.emplace_back(convert(&elseOp));
    }
  } // else scope deallocated here
  return ifStmt;
}

void FSMOperationConverter::convert(mlir::fsm::AddOp op) {
  symbolTable.insert(op.result(), symbolTable.lookup(op.lhs()) + detail::BinaryOps.plus.str() +
                                      symbolTable.lookup(op.rhs()));
}

nlohmann::json FSMOperationConverter::convert(mlir::fsm::SetAdd op) {
  auto st = op.theSet().getType().cast<SetType>();
  auto theSet = detail::Set{
      FSMConvertType(st.getElementType()),
      st.getNumElements()
  };
  return detail::ProcCall{
    theSet.set_add_fname(),
        {
          detail::ExprID{symbolTable.lookup(op.theSet())},
          detail::ExprID{symbolTable.lookup(op.value())}
      }
  };
}

std::string FSMConvertType(Type type) {
  if (auto stateType = type.dyn_cast<StateType>())
    return murphi::detail::e_cache_state_t();
  if (auto msgType = type.dyn_cast<MsgType>())
    return murphi::detail::r_message_t;
  if (auto dataType = type.dyn_cast<DataType>())
    return murphi::detail::ss_cache_val_t;
  if (auto rangeType = type.dyn_cast<RangeType>())
    return std::to_string(rangeType.getStart()) + ".." +
           std::to_string(rangeType.getEnd());
  if (auto idType = type.dyn_cast<IDType>())
    return detail::e_machines_t;
  assert(0 && "unsupported murphi type conversion");
}

} // namespace murphi