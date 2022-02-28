#include "translation/murphi/codegen/FSMDialectInterpreter.h"
#include "FSM/FSMOps.h"
#include "FSM/FSMUtils.h"
#include "translation/murphi/codegen/FSMOperationConverter.h"
#include "translation/murphi/codegen/MurphiConstants.h"
#include "translation/murphi/codegen/MurphiStructs.h"
#include <set>

using namespace mlir;
using namespace mlir::fsm;
using namespace nlohmann;
constexpr size_t default_reserve_amount = 10;

namespace {

json getMurphiMachineStatements(llvm::StringRef state, llvm::StringRef action,
                                llvm::StringRef machId, ModuleOp theModule) {
  auto transOp = theModule.lookupSymbol<MachineOp>(machId)
                     .lookupSymbol<StateOp>(state)
                     .lookupSymbol<TransitionOp>(action);
  if (!transOp)
    return nullptr;

  murphi::FSMOperationConverter opConverter;

  return opConverter.convert(transOp);
}

std::string mapFSMTypeToMurphiType(Type t) {
  if (t.isa<RangeType>()) {
    auto rt = t.cast<RangeType>();
    return std::to_string(rt.getStart()) + ".." + std::to_string(rt.getEnd());
  }

  if (t.isa<IDType>())
    return murphi::detail::e_machines_t;

  if (t.isa<DataType>())
    return murphi::detail::ss_cache_val_t;

  if (t.isa<MsgType>())
    return murphi::detail::r_message_t;

  assert(0 && "variable op has invalid type");
}
std::string mapFSMTypeToMurphiType(VariableOp var) {
  Type t = var.getType();
  if (t.isa<StateType>())
    return var->getParentOfType<MachineOp>().sym_name().str() +
           murphi::detail::state_suffix;
  if (t.isa<SetType>()) {
    auto st = t.cast<SetType>();
    return murphi::detail::Set{mapFSMTypeToMurphiType(st.getElementType()),
                               st.getNumElements()}
        .getSetId();
  }
  return mapFSMTypeToMurphiType(var.getType());
}

json convertMachineType(MachineOp mach) {
  std::vector<std::pair<std::string, std::string>> rec;
  for (auto varOp : mach.getOps<VariableOp>()) {
    auto entry =
        std::make_pair(varOp.name().str(), mapFSMTypeToMurphiType(varOp));
    rec.emplace_back(std::move(entry));
  }
  return murphi::detail::TypeDecl<murphi::detail::Record>{
      murphi::detail::EntryKey + mach.sym_name().str(), {std::move(rec)}};
}

} // namespace
namespace murphi {

FSMDialectInterpreter::FSMDialectInterpreter(ModuleOp m)
    : theModule{m},
      mTypeElems{
          {murphi::detail::c_adr,
           murphi::detail::ss_address_t}, // target address
          {murphi::detail::c_mtype,
           murphi::detail::e_message_type_t},                    // message type
          {murphi::detail::c_src, murphi::detail::e_machines_t}, // source
          {murphi::detail::c_dst, murphi::detail::e_machines_t}  // destination
      } {
  // Here we establish what additional fields have to be added to set of all
  // element types
  theModule.walk([&](MessageDecl msgDecl) {
    msgDecl.walk([&](MessageVariable var) {
      mTypeElems.emplace(
          std::make_pair(var.sym_name().str(), FSMConvertType(var.getType())));
    });
  });
}

json FSMDialectInterpreter::getMurphiCacheStatements(llvm::StringRef state,
                                                     llvm::StringRef action) {
  return getMurphiMachineStatements(
      translation::utils::demangleState(state.str()), action, "cache",
      theModule);
}

json FSMDialectInterpreter::getMurphiDirectoryStatements(
    llvm::StringRef state, llvm::StringRef action) {
  return getMurphiMachineStatements(
      translation::utils::demangleState(state.str()), action, "directory",
      theModule);
}

std::vector<std::string> FSMDialectInterpreter::getMessageNames() {
  std::set<std::string> messageNames;
  theModule.walk(
      [&](MessageOp op) { messageNames.insert(op.msgName().str()); });
  return {std::begin(messageNames), std::end(messageNames)};
}

/// Get the global Message type used in Murphi
json FSMDialectInterpreter::getMessageType() {
  std::vector<std::pair<std::string, std::string>> elems = {mTypeElems.begin(),
                                                            mTypeElems.end()};
  return murphi::detail::TypeDecl<murphi::detail::Record>{
      murphi::detail::r_message_t, {elems}};
}

std::vector<std::string> FSMDialectInterpreter::getMessageTypeNames() {
  std::vector<std::string> names;
  theModule.walk(
      [&names](MessageDecl decl) { names.push_back(decl.sym_name().str()); });
  return names;
}

nlohmann::json FSMDialectInterpreter::getMessageFactory(std::string &msgType) {
  auto definingMsg = theModule.lookupSymbol<MessageDecl>(msgType);
  assert(definingMsg != nullptr &&
         "Could not resolve the reference to the message");
  auto msgFactoryFunction = detail::GenericMurphiFunction{
      msgType,
      {},
      detail::ID{detail::r_message_t},
      {detail::ForwardDecl<detail::TypeDecl<detail::ID>>{
          "var", {detail::c_msg, {detail::r_message_t}}}},
      {}};

  // add the necessary parameters
  // adr : Address
  msgFactoryFunction.params.emplace_back(
      detail::Formal<detail::ID>{detail::c_adr, {detail::ss_address_t}});
  // mtype : MessageType
  msgFactoryFunction.params.emplace_back(
      detail::Formal<detail::ID>{detail::c_mtype, {detail::e_message_type_t}});
  // src : Machines
  msgFactoryFunction.params.emplace_back(
      detail::Formal<detail::ID>{detail::c_src, {detail::e_machines_t}});
  // dst :: Machines
  msgFactoryFunction.params.emplace_back(
      detail::Formal<detail::ID>{detail::c_dst, {detail::e_machines_t}});

  /// for each additional variable in the type add an additional parameter
  for (auto msgVar : definingMsg.getOps<MessageVariable>()) {
    msgFactoryFunction.params.emplace_back(detail::Formal<detail::ID>{
        msgVar.sym_name().str(), {FSMConvertType(msgVar.getType())}});
  }

  /// Now we can setup the statements
  for (auto r_msg_elem : mTypeElems) {
    std::string valToBeAssigned;
    auto is_default_param = [&](const std::string &field) {
      return field == detail::c_adr || field == detail::c_mtype ||
             field == detail::c_src || field == detail::c_dst;
    };
    if (definingMsg.lookupSymbol(r_msg_elem.first) != nullptr ||
        is_default_param(r_msg_elem.first)) {
      valToBeAssigned = r_msg_elem.first;
    } else {
      valToBeAssigned = detail::c_undef;
    }

    auto asin = detail::Assignment<detail::Designator, detail::ExprID>{
        {detail::c_msg,
         {detail::Indexer{"object", detail::ExprID{r_msg_elem.first}}}},
        {valToBeAssigned}};
    msgFactoryFunction.statements.emplace_back(std::move(asin));
  }

  // return the message
  msgFactoryFunction.statements.emplace_back(
      detail::ReturnStmt{detail::ExprID{detail::c_msg}});

  return msgFactoryFunction;
}

std::vector<std::string> FSMDialectInterpreter::getCacheStateNames() {
  std::vector<std::string> stateNames;
  stateNames.reserve(default_reserve_amount);

  std::vector<StateOp> allCacheStateOps;
  allCacheStateOps.reserve(default_reserve_amount);
  Operation *theCache = theModule.lookupSymbol("cache");
  utils::searchFor<StateOp>(theCache, allCacheStateOps);

  for (auto stateOp : allCacheStateOps) {
    stateNames.push_back(
        translation::utils::mangleCacheState(stateOp.sym_name().str()));
  }

  return stateNames;
}

std::vector<std::string> FSMDialectInterpreter::getDirectoryStateNames() {
  std::vector<std::string> stateNames;
  stateNames.reserve(default_reserve_amount);

  std::vector<StateOp> allDirectoryStates;
  allDirectoryStates.reserve(default_reserve_amount);
  Operation *theCache = theModule.lookupSymbol("directory");
  utils::searchFor<StateOp>(theCache, allDirectoryStates);

  for (auto stateOp : allDirectoryStates) {
    stateNames.push_back(
        translation::utils::mangleDirectoryState(stateOp.sym_name().str()));
  }

  return stateNames;
}

std::vector<std::string> FSMDialectInterpreter::getCacheStableStateNames() {
  std::vector<StateOp> stableStates;
  utils::searchForIf(theModule, stableStates, [](StateOp stateOp) {
    return !utils::isTransientState(stateOp);
  });

  std::set<std::string> outs;
  std::for_each(
      std::begin(stableStates), std::end(stableStates), [&outs](StateOp op) {
        outs.insert(translation::utils::mangleCacheState(op.sym_name().str()));
      });
  return {outs.begin(), outs.end()};
}

json FSMDialectInterpreter::getSetTypes() {
  json data;
  theModule.walk([&](VariableOp varOp) {
    if (varOp.getType().isa<SetType>()) {
      auto st = varOp.getType().cast<SetType>();
      auto theSet = detail::Set{mapFSMTypeToMurphiType(st.getElementType()),
                                st.getNumElements()};
      setTypeMap.insert({st, theSet});
      // push back the decls
      json jsonified = theSet;
      for (auto &j : jsonified)
        data.push_back(j);
    }
  });
  return data;
}

nlohmann::json FSMDialectInterpreter::getSetOperationImpl() {
  json data = json::array();
  for (auto &elem : setTypeMap) {
    auto theSet = elem.second;
    // .add()
    data.push_back(detail::SetAdd{theSet});
    // .count()
    data.push_back(detail::SetCount{theSet});
    // .contains()
    data.push_back(detail::SetContains{theSet});
    // .del()
    data.push_back(detail::SetDelete{theSet});
    // .clear()
    data.push_back(detail::SetClear{theSet});
  }
  return data;
}

json FSMDialectInterpreter::getCacheType() {
  auto theCache = theModule.lookupSymbol<MachineOp>("cache");
  return convertMachineType(theCache);
}

json FSMDialectInterpreter::getDirectoryType() {
  auto dir = theModule.lookupSymbol<MachineOp>("directory");
  return convertMachineType(dir);
}

std::vector<std::pair<std::string, std::string>>
FSMDialectInterpreter::getNetworks() {
  std::vector<std::pair<std::string, std::string>> networks;
  for (auto netOp : theModule.getOps<NetworkOp>()) {
    networks.emplace_back(std::make_pair(
        netOp.sym_name().getLeafReference().str(), netOp.ordering().str()));
  }
  return networks;
}

std::string compute_default_value(VariableOp varOp) {
  auto theMach = varOp->getParentOfType<MachineOp>();
  auto manglePref = theMach.sym_name().str() == "directory"
                        ? detail::directory_state_prefix
                        : detail::cache_state_prefix;
  if (varOp.initValue().hasValue()) {
    auto initAttr = varOp.initValueAttr();
    if (initAttr.isa<StringAttr>()) {
      auto attr = initAttr.cast<StringAttr>();
      // special case for state types -> we need to mangle
      auto init = attr.getValue().str();
      if (varOp.getType().isa<StateType>()) {
        return translation::utils::mangleState(init, manglePref);
      }
      return init;
    }
  }
  return "0";
}

nlohmann::json compute_mach_ss(const std::string &machId, MachineOp theMach) {
  constexpr auto mach_idx = "i";
  constexpr auto adr_idx = "a";
  auto for_mach = detail::ForStmt<detail::ForEachQuantifier<detail::ID>>{
      {mach_idx, {detail::SetKey + machId}}, {}};

  auto for_adr = detail::ForStmt<detail::ForEachQuantifier<detail::ID>>{
      {adr_idx, {detail::ss_address_t}}, {}};

  // i_cache[i][a].??? := cache_I;
  auto common_start =
      detail::Designator{detail::mach_prefix_v + machId,
                         {detail::Indexer{"array", detail::ExprID{mach_idx}},
                          detail::Indexer{"array", detail::ExprID{adr_idx}}}};

  for (auto varOp : theMach.getOps<VariableOp>()) {
    Type varOpType = varOp.getType();
    // dont instantiate {id types or sets of id types} types
    if (!varOpType.isa<IDType>()) {
      auto lhs = common_start; // copy the common_start
      lhs.indexes.emplace_back(
          detail::Indexer{"object", detail::ExprID{varOp.name().str()}});
      std::string defaultValue = compute_default_value(varOp);
      for_adr.stmts.emplace_back(
          detail::Assignment<decltype(lhs), detail::ExprID>{lhs,
                                                            {defaultValue}});
    }
  }
  for_mach.stmts.emplace_back(std::move(for_adr));
  return for_mach;
}

nlohmann::json FSMDialectInterpreter::getCacheStartState() {
  return compute_mach_ss("cache", theModule.lookupSymbol<MachineOp>("cache"));
}

nlohmann::json FSMDialectInterpreter::getDirectoryStartState() {
  return compute_mach_ss("directory",
                         theModule.lookupSymbol<MachineOp>("directory"));
}

bool FSMDialectInterpreter::isMulticastEnabled() {
  std::vector<MulticastOp> mcastOps;
  utils::searchFor(theModule.getOperation(), mcastOps);
  return !mcastOps.empty();
}

} // namespace murphi
