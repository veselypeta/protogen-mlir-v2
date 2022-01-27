#include "translation/murphi/codegen/MurphiCodeGen.h"
#include "translation/utils/JSONValidation.h"
#include <iostream>

using namespace inja;

namespace murphi {
/*
 * Implementation Details
 */
namespace detail {

bool validateMurphiJSON(const json &j) {
  const std::string schema_path =
      std::string(JSONValidation::schema_base_directory) +
      "gen_Murphi_json_schema.json";
  return JSONValidation::validate_json(schema_path, j);
}

} // namespace detail

mlir::LogicalResult MurphiCodeGen::translate() {
  generateConstants();
  generateTypes();
  generateVars();
  generateMethods();
  generateRules();
  return render();
}

bool MurphiCodeGen::is_json_valid() { return detail::validateMurphiJSON(data); }

void MurphiCodeGen::generateConstants() {
  // add the defined constants
  for (auto constantOp : moduleInterpreter.getConstants()) {
    data["decls"]["const_decls"].push_back(
        detail::ConstDecl{constantOp.id().str(), constantOp.val()});
  }

  // boilerplate constants
  data["decls"]["const_decls"].push_back(
      detail::ConstDecl{detail::c_val_cnt_t, detail::c_val_max});
  data["decls"]["const_decls"].push_back(
      detail::ConstDecl{detail::c_adr_cnt_t, detail::c_adr_cnt});
  data["decls"]["const_decls"].push_back(
      detail::ConstDecl{detail::c_ordered_t, detail::c_ordered_size});
  data["decls"]["const_-decls"].push_back(
      detail::ConstDecl{detail::c_unordered_t, detail::c_unordered_size});
}

void MurphiCodeGen::generateTypes() {
  _typeEnums();
  _typeStatics();
  _typeMachineSets();
  _typeMessage();
  _typeMachines();
  _typeNetworkObjects();
  _typeMutexes();
}

mlir::LogicalResult MurphiCodeGen::render() {
  return renderMurphi(data, output);
}

void MurphiCodeGen::_typeEnums() {
  // access enum
  data["decls"]["type_decls"].push_back(detail::TypeDecl<detail::Enum>{
      detail::e_access_t, detail::Enum{{"none", "load", "store"}}});
  // messages enum
  data["decls"]["type_decls"].push_back(detail::TypeDecl<detail::Enum>{
      detail::e_message_type_t, {moduleInterpreter.getEnumMessageTypes()}});
  // cache state
  data["decls"]["type_decls"].push_back(detail::TypeDecl<detail::Enum>{
      detail::e_cache_state_t(),
      {moduleInterpreter.getEnumMachineStates(detail::machines.cache.str())}});
  data["decls"]["type_decls"].push_back(
      detail::TypeDecl<detail::Enum>{detail::e_directory_state_t(),
                                     {moduleInterpreter.getEnumMachineStates(
                                         detail::machines.directory.str())}});
}

void MurphiCodeGen::_typeStatics() {
  // Address type
  data["decls"]["type_decls"].push_back(
      detail::TypeDecl<detail::ScalarSet<std::string>>{
          detail::ss_address_t,
          detail::ScalarSet<std::string>{detail::c_val_cnt_t}});
  // ClValue type
  data["decls"]["type_decls"].push_back(
      detail::TypeDecl<detail::SubRange<size_t, std::string>>{
          detail::ss_cache_val_t, {0, detail::c_val_cnt_t}});
}

void MurphiCodeGen::_typeMachineSets() {
  auto add_mach_decl_type = [&](const std::string &machine,
                                const mlir::pcc::PCCType &type) {
    std::string typeIdent = machine == detail::machines.cache
                                ? detail::cache_set_t()
                                : detail::directory_set_t();
    if (type.isa<mlir::pcc::SetType>()) {
      auto setSize = type.cast<mlir::pcc::SetType>().getNumElements();
      data["decls"]["type_decls"].push_back(
          detail::TypeDecl<detail::ScalarSet<size_t>>{
              detail::cache_set_t(), detail::ScalarSet<size_t>{setSize}});
    } else {
      // must be a struct type
      assert(type.isa<mlir::pcc::StructType>() &&
             "machine was not of set/struct type!");
      data["decls"]["type_decls"].push_back(
          detail::TypeDecl<detail::Enum>{typeIdent, {{machine}}});
    }
  };

  // add the cache set
  add_mach_decl_type(detail::machines.cache.str(),
                     moduleInterpreter.getCache().getType());
  add_mach_decl_type(detail::machines.directory.str(),
                     moduleInterpreter.getDirectory().getType());

  // push a union of these for the Machines type
  data["decls"]["type_decls"].push_back(detail::TypeDecl<detail::Union>{
      detail::e_machines_t,
      {{detail::cache_set_t(), detail::directory_set_t()}}});
}

void MurphiCodeGen::_typeNetworkObjects() {
  for (const auto &type : boilerplate::emitNetworkDefinitionJson()) {
    data["decls"]["type_decls"].push_back(type);
  }
}

void MurphiCodeGen::_typeMachines() {
  // *** cache *** //
  mlir::pcc::CacheDeclOp cacheOp = moduleInterpreter.getCache();
  _getMachineEntry(cacheOp.getOperation());

  // *** directory *** //
  mlir::pcc::DirectoryDeclOp directoryOp = moduleInterpreter.getDirectory();

  _getMachineEntry(directoryOp.getOperation());
  _getMachineMach();
  _getMachineObjs();
}

void MurphiCodeGen::_getMachineEntry(mlir::Operation *machineOp) {
  // Check that the operation is either cache or directory decl
  const auto opIdent = machineOp->getName().getIdentifier().strref();
  assert((opIdent == detail::opStringMap.cache_decl ||
          opIdent == detail::opStringMap.dir_decl) &&
         "invalid operation passed to generation machine function");
  const auto machineIdent = opIdent == detail::opStringMap.cache_decl
                                ? detail::machines.cache
                                : detail::machines.directory;

  const auto is_id_attr = [](const mlir::NamedAttribute &attr) {
    return attr.first == "id";
  };

  std::vector<std::pair<std::string, std::string>> record_elems;

  const auto generate_mach_attr_field =
      [&record_elems, &machineIdent](const mlir::NamedAttribute &attr) {
        std::string fieldID = attr.first.str();
        mlir::TypeAttr typeAttr = attr.second.cast<mlir::TypeAttr>();
        std::string fieldType =
            MLIRTypeToMurphiTypeRef(typeAttr.getValue(), machineIdent);
        record_elems.emplace_back(fieldID, fieldType);
      };

  // for each attribute; generate a machine attribute
  std::for_each(machineOp->getAttrs().begin(), machineOp->getAttrs().end(),
                [&](const mlir::NamedAttribute &attribute) {
                  const mlir::NamedAttribute named_attr =
                      static_cast<mlir::NamedAttribute>(attribute);
                  if (!is_id_attr(named_attr)) {
                    generate_mach_attr_field(named_attr);
                  }
                });

  std::string machine_id_t = machineIdent == detail::machines.cache
                                 ? detail::r_cache_entry_t()
                                 : detail::r_directory_entry_t();
  // generate the correct murphi declaration
  data["decls"]["type_decls"].push_back(detail::TypeDecl<detail::Record>{
      std::move(machine_id_t), {std::move(record_elems)}});
}

void MurphiCodeGen::_getMachineMach() {

  detail::TypeDecl<detail::Array<detail::ID, detail::ID>> cacheMach{
      detail::cache_mach_t(),
      {{detail::ss_address_t}, {detail::cache_set_t()}}};

  detail::TypeDecl<detail::Array<detail::ID, detail::ID>> dirMach{
      detail::directory_mach_t(),
      {{detail::ss_address_t}, {detail::directory_set_t()}}};

  data["decls"]["type_decls"].push_back(std::move(cacheMach));
  data["decls"]["type_decls"].push_back(std::move(dirMach));
}

void MurphiCodeGen::_getMachineObjs() {

  detail::TypeDecl<detail::Array<detail::ID, detail::ID>> cacheObj{
      detail::cache_obj_t(),
      {{detail::cache_set_t()}, {detail::cache_mach_t()}}};

  detail::TypeDecl<detail::Array<detail::ID, detail::ID>> dirObj{
      detail::directory_obj_t(),
      {{detail::directory_set_t()}, {detail::directory_mach_t()}}};

  data["decls"]["type_decls"].push_back(std::move(cacheObj));
  data["decls"]["type_decls"].push_back(std::move(dirObj));
}

void MurphiCodeGen::_typeMessage() {
  // generate the glob msg
  data["decls"]["type_decls"].push_back(detail::TypeDecl<detail::Record>{
      detail::r_message_t, {get_glob_msg_type()}});
}

void MurphiCodeGen::_typeMutexes() {
  detail::TypeDecl<detail::Array<detail::ID, detail::ID>> mutex_t{
      detail::a_cl_mutex_t, {{detail::ss_address_t}, {"boolean"}}};
  data["decls"]["type_decls"].push_back(std::move(mutex_t));
}

std::vector<std::pair<std::string, std::string>>
MurphiCodeGen::get_glob_msg_type() {
  std::map<std::string, std::string> glob_msg_type;
  auto all_msg_types = moduleInterpreter.getMessages();

  std::for_each(
      all_msg_types.begin(), all_msg_types.end(),
      [&](const mlir::pcc::MsgDeclOp &msgDeclOp) {
        std::for_each(
            msgDeclOp->getAttrs().begin(), msgDeclOp->getAttrs().end(),
            [&](const mlir::NamedAttribute &named_attr) {
              if (named_attr.first != "id") { // skip the id field
                mlir::TypeAttr typeAttr =
                    named_attr.second.cast<mlir::TypeAttr>();
                glob_msg_type.insert(
                    {named_attr.first.str(),
                     MLIRTypeToMurphiTypeRef(typeAttr.getValue(), "")});
              }
            });
      });

  // copy to vector
  std::vector<std::pair<std::string, std::string>> msg_decl_type;
  msg_decl_type.reserve(glob_msg_type.size() + 1);
  for (auto &mv : glob_msg_type) {
    msg_decl_type.emplace_back(mv);
  }

  // push back the address type also
  msg_decl_type.emplace_back(detail::c_adr, detail::ss_address_t);

  return msg_decl_type;
}

// Free Functions
std::string MLIRTypeToMurphiTypeRef(const mlir::Type &t,
                                    const llvm::StringRef mach) {
  if (t.isa<mlir::pcc::DataType>()) {
    return detail::ss_cache_val_t;
  }
  if (t.isa<mlir::pcc::StateType>()) {
    return mach == detail::machines.cache ? detail::e_cache_state_t()
                                          : detail::e_directory_state_t();
  }
  if (t.isa<mlir::pcc::IDType>()) {
    return detail::e_machines_t;
  }
  if (t.isa<mlir::pcc::MsgIdType>()) {
    return detail::r_message_t;
  }
  // TODO - fix this implementation
  // we're better off if we declare a type for this first
  // and then refer to it.
  if (t.isa<mlir::pcc::SetType>()) {
    // create a type that we can refer to. This may be more challenging now
    mlir::pcc::SetType st = t.cast<mlir::pcc::SetType>();

    return "multiset [ " + std::to_string(st.getNumElements()) + " ] of " +
           MLIRTypeToMurphiTypeRef(st.getElementType(), "");
  }

  if (t.isa<mlir::pcc::IntRangeType>()) {
    mlir::pcc::IntRangeType intRangeType = t.cast<mlir::pcc::IntRangeType>();
    return std::to_string(intRangeType.getStartRange()) + ".." +
           std::to_string(intRangeType.getEndRange());
  }

  // TODO - add support for more types
  assert(0 && "currently using an unsupported type!");
}

void MurphiCodeGen::generateVars() {
  _varMachines();
  _varNetworks();
  _varMutexes();
}

void MurphiCodeGen::_varMachines() {

  detail::VarDecl<detail::ID> var_cache{detail::cache_v(),
                                        {{detail::cache_obj_t()}}};
  detail::VarDecl<detail::ID> var_dir{detail::directory_v(),
                                      {{detail::directory_obj_t()}}};
  data["decls"]["var_decls"].push_back(var_cache);
  data["decls"]["var_decls"].push_back(var_dir);
}
void MurphiCodeGen::_varNetworks() {
  auto ops = moduleInterpreter.getOperations<mlir::pcc::NetDeclOp>();
  std::for_each(ops.begin(), ops.end(), [&](mlir::pcc::NetDeclOp &netDeclOp) {
    if (netDeclOp.getType().getOrdering() == "ordered") {

      detail::VarDecl<detail::ID> ord_net_v{
          netDeclOp.netId().str(),
          {{std::string(detail::ObjKey) + detail::ordered}}};
      detail::VarDecl<detail::ID> ord_net_cnt_v{
          std::string(detail::CntKey) + netDeclOp.netId().str(),
          {{std::string(detail::ObjKey) + detail::ordered_cnt}}};
      data["decls"]["var_decls"].push_back(ord_net_v);
      data["decls"]["var_decls"].push_back(ord_net_cnt_v);

    } else {
      detail::VarDecl<detail::ID> unord_net_v{
          netDeclOp.netId().str(),
          {{std::string(detail::ObjKey) + detail::unordered}}};
      data["decls"]["var_decls"].push_back(unord_net_v);
    }
  });
}
void MurphiCodeGen::_varMutexes() {
  detail::VarDecl<detail::ID> mutex_v{detail::cl_mutex_v,
                                      {{detail::a_cl_mutex_t}}};
  data["decls"]["var_decls"].push_back(mutex_v);
}

/*
 * Murphi Methods
 */

void MurphiCodeGen::generateMethods() {
  _generateMsgFactories();
  _generateHelperFunctions();
}

/*
 * MESSAGE FACTORIES
 */

void MurphiCodeGen::_generateMsgFactories() {
  auto msgDecls = moduleInterpreter.getOperations<mlir::pcc::MsgDeclOp>();
  std::for_each(std::begin(msgDecls), std::end(msgDecls),
                [&](mlir::pcc::MsgDeclOp msgDeclOp) {
                  data["proc_decls"].push_back(
                      {{"procType", "function"},
                       {"def", detail::MessageFactory{msgDeclOp}}});
                });
}

/*
 * Helper Functions
 */

void MurphiCodeGen::_generateHelperFunctions() {
  _generateMutexHelpers();
  _generateSendPopFunctions();
  _generateMachineHandlers();
  _generateCPUEventHandlers();
}

void MurphiCodeGen::_generateMutexHelpers() {
  constexpr char adr_param[] = "adr";

  // ACQUIRE MUTEX
  json acq_mut_proc = {
      {"id", detail::aq_mut_f},
      {"params", json::array({detail::Formal<detail::ID>{
                     adr_param, {{detail::ss_address_t}}}})},
      {"statements",
       json::array({detail::Assignment<detail::Designator, detail::ExprID>{
           {detail::cl_mutex_v,
            {detail::Indexer{"array", detail::ExprID{adr_param}}}},
           {"true"}}

       })}};
  json acq_proc_decl = {{"procType", "procedure"},
                        {"def", std::move(acq_mut_proc)}};

  // RELEASE MUTEX
  json rel_mut_proc = {
      {"id", detail::rel_mut_f},
      {"params", json::array({detail::Formal<detail::ID>{
                     adr_param, {{detail::ss_address_t}}}})},
      {"statements",
       json::array({detail::Assignment<detail::Designator, detail::ExprID>{
           {detail::cl_mutex_v,
            {detail::Indexer{"array", detail::ExprID{adr_param}}}},
           {"false"}}

       })}};
  json rel_proc_decl = {{"procType", "procedure"},
                        {"def", std::move(rel_mut_proc)}};

  // Add them to the json structure
  data["proc_decls"].push_back(std::move(acq_proc_decl));
  data["proc_decls"].push_back(std::move(rel_proc_decl));
}

void MurphiCodeGen::_generateSendPopFunctions() {
  auto networks = moduleInterpreter.getNetworks();
  std::for_each(
      networks.begin(), networks.end(), [&](mlir::pcc::NetDeclOp netDeclOp) {
        auto netID = netDeclOp.netId();
        auto order = netDeclOp.result()
                         .getType()
                         .cast<mlir::pcc::NetworkType>()
                         .getOrdering();
        if (order == "ordered") {
          // ordered networks
          data["proc_decls"].push_back(
              detail::OrderedSendFunction{netID.str()});
          data["proc_decls"].push_back(detail::OrderedPopFunction{netID.str()});
        } else {
          data["proc_decls"].push_back(
              detail::UnorderedSendFunction{netID.str()});
        }
      });
}

/*
 * Machine Handlers
 */

void MurphiCodeGen::_generateMachineHandlers() {
  // cache
  // TODO - generate the inner states

  auto cacheHandler = detail::MachineHandler{detail::machines.cache.str(), {}};
  data["proc_decls"].push_back(cacheHandler);

  // directory
  // TODO - generate directory inner states
  auto directoryHandler =
      detail::MachineHandler{detail::machines.directory.str(), {}};
  data["proc_decls"].push_back(directoryHandler);
}

/*
 * CPU Event Handler
 */

void MurphiCodeGen::_generateCPUEventHandlers() {
  // For Each stable state generate CPU Event handlers for load/store/evict
  // I -> Special state -> NO EVICT
  auto stableStates = std::vector<std::string>{"cache_I", "cache_S", "cache_M"};

  for (auto &ss : stableStates) {
    for (auto &cpu_event : detail::cpu_events) {
      if (cpu_event != "evict" || ss != "cache_I") {
        auto cpu_event_handler =
            detail::CPUEventHandler{ss + "_" + cpu_event.str(), {}};
        data["proc_decls"].push_back(cpu_event_handler);
      }
    }
  }
}

/*
 * Rules
 */

void MurphiCodeGen::generateRules() {
  _generateCacheRuleHandler();
  _generateNetworkRules();
  _generateStartState();
}

void MurphiCodeGen::_generateCacheRuleHandler() {
  detail::CacheRuleHandler cacheRuleHandler;
  const auto stableStates =
      std::vector<std::string>{"cache_I", "cache_S", "cache_M"};

  for (auto &ss : stableStates) {
    for (auto cpu_event : detail::cpu_events) {
      if (cpu_event != "evict" || ss != "cache_I") {
        auto cpuEventRule = detail::CPUEventRule{ss, cpu_event.str()};
        cacheRuleHandler.rules.emplace_back(std::move(cpuEventRule));
      }
    }
  }
  data["rules"].push_back(std::move(cacheRuleHandler));
}

void MurphiCodeGen::_generateNetworkRules() {
  auto networks = moduleInterpreter.getNetworks();
  for (auto &nw : networks) {
    if (nw.getType().getOrdering() == "ordered") {
      data["rules"].push_back(detail::OrderedRuleset{nw.netId().str()});
    } else {
      data["rules"].push_back(detail::UnorderedRuleset{nw.netId().str()});
    }
  }
}

void MurphiCodeGen::_generateStartState() {
  auto ss = detail::StartState{"murphi start state", {}, {}};
  ///*** Initialize the Directory and Cache objects ***///
  auto mach_ss_stmts = [&](const std::string &machId) -> json {
    constexpr auto mach_idx = "i";
    constexpr auto adr_idx = "a";
    auto for_mach = detail::ForStmt<detail::ForEachQuantifier<detail::ID>>{
        {mach_idx, {detail::SetKey + machId}}, {}};

    auto for_adr = detail::ForStmt<detail::ForEachQuantifier<detail::ID>>{
        {adr_idx, {detail::ss_address_t}}, {}};

    // for each value in the struct setup a default;
    // i_cache[i][a].??? := cache_I;
    // TODO - implement this properly
    auto decl = detail::ExprID{"State"};
    auto default_value = detail::ExprID{machId + "_I"};

    auto common_start =
        detail::Designator{detail::mach_prefix_v + machId,
                           {detail::Indexer{"array", detail::ExprID{mach_idx}},
                            detail::Indexer{"array", detail::ExprID{adr_idx}}}};

    auto lhs = common_start; // copy the start
    lhs.indexes.emplace_back(detail::Indexer{"object", detail::ExprID{decl}});

    for_adr.stmts.emplace_back(
        detail::Assignment<decltype(lhs), decltype(default_value)>{
            lhs, default_value});
    for_mach.stmts.emplace_back(for_adr);
    return for_mach;
  };
  ss.statements.push_back(mach_ss_stmts(detail::machines.cache.str()));
  ss.statements.push_back(mach_ss_stmts(detail::machines.directory.str()));

  /// *** Initialize Mutexes *** ///
  auto generate_mutex_inits = []() -> json {
    constexpr auto adr_idx = "a";
    auto mut_false = detail::Assignment<detail::Designator, detail::ExprID>{
        {detail::cl_mutex_v,
         {detail::Indexer{"array", detail::ExprID{adr_idx}}}},
        {"false"}};
    auto for_adr = detail::ForStmt<detail::ForEachQuantifier<detail::ID>>{
        {adr_idx, {detail::ss_address_t}}, {std::move(mut_false)}};
    return for_adr;
  };
  ss.statements.emplace_back(generate_mutex_inits());

  /// ** Undefine networks //
  for (auto nw : moduleInterpreter.getNetworks()) {
    auto netId = nw.netId().str();
    auto rhs = detail::Designator{netId, {}};
    ss.statements.emplace_back(detail::UndefineStmt<decltype(rhs)>{rhs});
  }

  /// ***  Set all ordered counts to zero  *** ///
  auto gen_ordered_cnt_ss = [](llvm::StringRef netId) -> json {
    constexpr auto mach_idx = "n";
    auto for_quant =
        detail::ForEachQuantifier<detail::ID>{mach_idx, {detail::e_machines_t}};

    auto cnt_stmt = detail::Assignment<detail::Designator, detail::ExprID>{
        {detail::CntKey + netId.str(),
         {detail::Indexer{"array", detail::ExprID{mach_idx}}}},
        {"0"}};

    return detail::ForStmt<decltype(for_quant)>{std::move(for_quant),
                                                {std::move(cnt_stmt)}};
  };

  for (auto &nw : moduleInterpreter.getNetworks()) {
    if (nw.getType().getOrdering() == "ordered") {
      ss.statements.emplace_back(gen_ordered_cnt_ss(nw.netId()));
    }
  }

  data["rules"].push_back(ss);
}

mlir::LogicalResult renderMurphi(const json &data, llvm::raw_ostream &output) {
  // validate json
  assert(detail::validateMurphiJSON(data) &&
         "JSON from codegen does not validate with the json schema");
  auto &env = InjaEnvSingleton::getInstance();
  auto tmpl = env.parse_template("murphi_base.tmpl");
  auto result = env.render(tmpl, data);
  output << result;
  return mlir::success();
}
} // namespace murphi
