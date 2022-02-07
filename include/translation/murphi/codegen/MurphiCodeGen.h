#pragma once
#include "inja/inja.hpp"
#include "mlir/IR/BuiltinDialect.h"
#include "translation/murphi/codegen/Boilerplate.h"
#include "translation/murphi/codegen/InjaEnvSingleton.h"
#include "translation/murphi/codegen/MurphiConstants.h"
#include "translation/murphi/codegen/MurphiStructs.h"
#include "translation/utils/ModuleInterpreter.h"
#include <type_traits>

// Forward Declarations
namespace murphi {
namespace detail {
struct ID;
}
} // namespace murphi

// For a manual on the Murphi Programming Language :
// https://www.cs.ubc.ca/~ajh/courses/cpsc513/assign-token/User.Manual
namespace murphi {
namespace detail {
bool validateMurphiJSON(const nlohmann::json &j);
} // namespace detail
/*
 * Murphi Translation Class
 */
class MurphiCodeGen {
public:
  MurphiCodeGen(mlir::ModuleOp op, llvm::raw_ostream &output)
      : moduleInterpreter{ModuleInterpreter{op}}, output{output} {
    // initialize decls
    data["decls"] = nlohmann::json::object();
    data["proc_decls"] = nlohmann::json::array();
    data["rules"] = nlohmann::json::array();
  }

  mlir::LogicalResult translate();

  // validate json
  bool is_json_valid();

  mlir::LogicalResult render();
  // *** Block generation
  void generateConstants();
  void generateTypes();
  void generateVars();

  // ** methods
  void generateMethods();

  // ** Rules
  void generateRules();

private:
  /*
   * Type Declarations functions
   */
  void _typeEnums();       // generate access/msgtype/states enums
  void _typeStatics();     // address space/ cl
  void _typeMachineSets(); // set cache/directory
  void _typeMessage();     // glob msg
  void _typeMachines();
  void _typeNetworkObjects();
  // additional helpers
  void _getMachineEntry(mlir::Operation *machineOp);
  void _getMachineMach();
  void _getMachineObjs();
  void _typeMutexes();

  std::vector<std::pair<std::string, std::string>> get_glob_msg_type();

  /*
   * Var Declaration functions
   */
  void _varMachines();
  void _varNetworks();
  void _varMutexes();

  /*
   * Methods
   */
  void _generateMsgFactories();
  void _generateHelperFunctions();
  void _generateMutexHelpers();
  void _generateSendPopFunctions();

  void _generateMachineHandlers();
  void _generateCPUEventHandlers();

  /*
   * Rules
   */
  void _generateCacheRuleHandler();
  void _generateNetworkRules();
  void _generateStartState();

  ModuleInterpreter moduleInterpreter;
  llvm::raw_ostream &output;
  nlohmann::json data;
};

// Free Functions
std::string MLIRTypeToMurphiTypeRef(const mlir::Type &t, llvm::StringRef mach);

mlir::LogicalResult renderMurphi(const nlohmann::json &data,
                                 llvm::raw_ostream &output);

template <class InterpreterT> class MurphiAssembler {
public:
  explicit MurphiAssembler(InterpreterT interpreter)
      : interpreter{std::move(interpreter)} {};

  nlohmann::json assemble() {
    nlohmann::json data = {{"decls",
                            {{"const_decls", nlohmann::json::array()},
                             {"type_decls", nlohmann::json::array()},
                             {"var_decls", nlohmann::json::array()}}},
                           {"proc_decls", nlohmann::json::array()},
                           {"rules", nlohmann::json::array()}};
    assembleDecls(data);
    assembleProcedures(data);
    assembleRules(data);
    return data;
  }

private:
  void assembleConstants(nlohmann::json &data) {
    murphi::boilerplate::setBoilerplateConstants(data);
    data["decls"]["const_decls"].push_back(
        detail::ConstDecl{detail::c_nr_cache, interpreter.getCacheSetSize()});
  }

  void assembleEnums(nlohmann::json &data) {
    // Access Type
    murphi::boilerplate::setBoilerplateAccessType(data);

    // Message Type
    data["decls"]["type_decls"].push_back(detail::TypeDecl<detail::Enum>{
        murphi::detail::e_message_type_t, {interpreter.getMessageNames()}});

    // Cache State
    data["decls"]["type_decls"].push_back(detail::TypeDecl<detail::Enum>{
        murphi::detail::e_cache_state_t(), {interpreter.getCacheStateNames()}});

    // Directory State
    data["decls"]["type_decls"].push_back(
        detail::TypeDecl<detail::Enum>{murphi::detail::e_directory_state_t(),
                                       {interpreter.getDirectoryStateNames()}});
  }

  void assembleMachineEntryTypes(nlohmann::json &data) {
    data["decls"]["type_decls"].emplace_back(interpreter.getCacheType());
    data["decls"]["type_decls"].emplace_back(interpreter.getDirectoryType());
  }

  void assembleStaticTypes(nlohmann::json &data) {
    // Address type
    data["decls"]["type_decls"].emplace_back(
        detail::TypeDecl<detail::ScalarSet<std::string>>{
            detail::ss_address_t,
            detail::ScalarSet<std::string>{detail::c_val_cnt_t}});
    // ClValue type
    data["decls"]["type_decls"].emplace_back(
        detail::TypeDecl<detail::SubRange<size_t, std::string>>{
            detail::ss_cache_val_t, {0, detail::c_val_cnt_t}});
  }

  void assembleSetTypes(nlohmann::json &data){
    for(nlohmann::json &set : interpreter.getSetTypes()){
      data["decls"]["type_decls"].emplace_back(set);
    }
  }

  void assembleMachineSets(nlohmann::json &data) {
    auto add_mach_decl_type = [&](const std::string &machine) {
      if (machine == detail::machines.cache) {
        data["decls"]["type_decls"].push_back(
            detail::TypeDecl<detail::ScalarSet<std::string>>{
                detail::cache_set_t(),
                detail::ScalarSet<std::string>{detail::c_nr_cache}});
      } else {
        // must be a directory
        data["decls"]["type_decls"].push_back(detail::TypeDecl<detail::Enum>{
            detail::directory_set_t(), {{machine}}});
      }
    };

    // add the cache set
    add_mach_decl_type(detail::machines.cache.str());
    add_mach_decl_type(detail::machines.directory.str());

    // push a union of these for the Machines type
    data["decls"]["type_decls"].push_back(detail::TypeDecl<detail::Union>{
        detail::e_machines_t,
        {{detail::cache_set_t(), detail::directory_set_t()}}});

    // Counts for ranges
    assembleSetTypes(data);
  }

  void assembleMultipleAddresses(nlohmann::json &data) {
    detail::TypeDecl<detail::Array<detail::ID, detail::ID>> cacheMach{
        detail::cache_mach_t(),
        {{detail::ss_address_t}, {detail::r_cache_entry_t()}}};

    detail::TypeDecl<detail::Array<detail::ID, detail::ID>> dirMach{
        detail::directory_mach_t(),
        {{detail::ss_address_t}, {detail::r_directory_entry_t()}}};
    data["decls"]["type_decls"].push_back(std::move(cacheMach));
    data["decls"]["type_decls"].push_back(std::move(dirMach));
  }

  void assembleMachineObjects(nlohmann::json &data) {

    detail::TypeDecl<detail::Array<detail::ID, detail::ID>> cacheObj{
        detail::cache_obj_t(),
        {{detail::cache_set_t()}, {detail::cache_mach_t()}}};

    detail::TypeDecl<detail::Array<detail::ID, detail::ID>> dirObj{
        detail::directory_obj_t(),
        {{detail::directory_set_t()}, {detail::directory_mach_t()}}};

    data["decls"]["type_decls"].push_back(std::move(cacheObj));
    data["decls"]["type_decls"].push_back(std::move(dirObj));
  }

  void assembleNetworkObjects(nlohmann::json &data) {
    for (const auto &type : boilerplate::emitNetworkDefinitionJson()) {
      data["decls"]["type_decls"].push_back(type);
    }
  }

  void assembleTypes(nlohmann::json &data) {
    assembleEnums(data);
    // assemble setup types
    assembleStaticTypes(data);

    // set types
    assembleMachineSets(data);

    // Message Type
    data["decls"]["type_decls"].emplace_back(interpreter.getMessageType());
    assembleMachineEntryTypes(data);

    // Machine Multiple Addresses
    assembleMultipleAddresses(data);

    // Assemble Machine Objects
    assembleMachineObjects(data);

    // Assemble Network Objects
    assembleNetworkObjects(data);
  }

  void assembleMachineVars(nlohmann::json &data) {

    detail::VarDecl<detail::ID> var_cache{detail::cache_v(),
                                          {{detail::cache_obj_t()}}};
    detail::VarDecl<detail::ID> var_dir{detail::directory_v(),
                                        {{detail::directory_obj_t()}}};
    data["decls"]["var_decls"].push_back(var_cache);
    data["decls"]["var_decls"].push_back(var_dir);
  }

  void assembleNetworkVars(nlohmann::json &data) {
    for (std::pair<std::string, std::string> &network :
         interpreter.getNetworks()) {
      std::string netId = network.first;
      std::string netOrder = network.second;
      if (netOrder == "ordered") {

        detail::VarDecl<detail::ID> ord_net_v{
            netId, {{std::string(detail::ObjKey) + detail::ordered}}};
        detail::VarDecl<detail::ID> ord_net_cnt_v{
            std::string(detail::CntKey) + netId,
            {{std::string(detail::ObjKey) + detail::ordered_cnt}}};
        data["decls"]["var_decls"].push_back(ord_net_v);
        data["decls"]["var_decls"].push_back(ord_net_cnt_v);

      } else {
        detail::VarDecl<detail::ID> unord_net_v{
            netId, {{std::string(detail::ObjKey) + detail::unordered}}};
        data["decls"]["var_decls"].push_back(unord_net_v);
      }
    }
  }

  void assembleVariables(nlohmann::json &data) {
    assembleMachineVars(data);
    assembleNetworkVars(data);
  }

  void assembleDecls(nlohmann::json &data) {
    assembleConstants(data);
    assembleTypes(data);
    assembleVariables(data);
  }

  void assembleMessageFactories(nlohmann::json &data) {
    for (std::string &msgName : interpreter.getMessageTypeNames())
      data["proc_decls"].emplace_back(interpreter.getMessageFactory(msgName));
  }

  void assembleNetworkSendFunctions(nlohmann::json &data) {
    for (std::pair<std::string, std::string> &network :
         interpreter.getNetworks()) {
      auto netID = network.first;
      auto order = network.second;
      if (order == "ordered") {
        // ordered networks
        data["proc_decls"].push_back(detail::OrderedSendFunction{netID});
        data["proc_decls"].push_back(detail::OrderedPopFunction{netID});
      } else {
        data["proc_decls"].push_back(detail::UnorderedSendFunction{netID});
      }
    }
  }

  void assembleSetHelperFunction(nlohmann::json &data){
    for(nlohmann::json &func : interpreter.getSetOperationImpl()){
      data["proc_decls"].push_back(func);
    }
  }

  template <class StateCallableT, class ConvertCallableT>
  void assembleMachineController(nlohmann::json &data,
                                 llvm::StringRef machineName,
                                 StateCallableT getStates,
                                 ConvertCallableT convert) {
    auto machineController = detail::MachineHandler{machineName.str(), {}};

    // for each available state
    for (const std::string &machState : getStates()) {
      auto stateCaseStmt = boilerplate::getBasicCaseStmt(machState);
      auto stateMsgSwitch = boilerplate::getMessageHandlerSwitch();

      // for each message name
      for (const std::string &msgName : interpreter.getMessageNames()) {

        nlohmann::json msgImpl = convert(machState, msgName);
        if (msgImpl == nullptr)
          continue;

        // construct the msg case
        auto msgCase =
            detail::CaseStmt{detail::ExprID{msgName}, std::move(msgImpl)};

        // add it to the current state handler
        stateMsgSwitch.cases.emplace_back(std::move(msgCase));
      }

      // add the stateMsgSwitch to the state case
      stateCaseStmt.statements = {stateMsgSwitch};
      // push back the state handler to the machine controller
      machineController.cases.emplace_back(stateCaseStmt);
    }

    // write the handler to the data
    data["proc_decls"].push_back(std::move(machineController));
  }

  void assembleDirectoryController(nlohmann::json &data) {
    assembleMachineController(
        data, detail::machines.directory,
        [&]() { return interpreter.getDirectoryStateNames(); },
        [&](llvm::StringRef state, llvm::StringRef action) {
          return interpreter.getMurphiDirectoryStatements(state, action);
        });
  }

  void assembleCacheController(nlohmann::json &data) {
    assembleMachineController(
        data, detail::machines.cache,
        [&]() { return interpreter.getCacheStateNames(); },
        [&](llvm::StringRef state, llvm::StringRef action) {
          return interpreter.getMurphiCacheStatements(state, action);
        });
  }

  void assembleStartStateFunctions(nlohmann::json &data) {
    std::vector<std::string> stableStates =
        interpreter.getCacheStableStateNames();

    for (auto stableState : stableStates) {
      for (auto event : murphi::detail::cpu_events) {

        nlohmann::json eventStatements =
            interpreter.getMurphiCacheStatements(stableState, event);
        if (eventStatements == nullptr)
          continue;

        data["proc_decls"].template emplace_back(detail::CPUEventHandler{
            stableState + "_" + event.str(), std::move(eventStatements)});
      }
    }
  }

  void assembleProcedures(nlohmann::json &data) {
    assembleMessageFactories(data);
    assembleNetworkSendFunctions(data);
    assembleSetHelperFunction(data);
    assembleCacheController(data);
    assembleDirectoryController(data);
    assembleStartStateFunctions(data);
  }

  void assembleRules(nlohmann::json &data) {
    assembleCacheRuleset(data);
    assembleNetworkRulesets(data);
    assembleStartStateRules(data);
    assembleInvariant(data);
  }

  void assembleCacheRuleset(nlohmann::json &data) {
    murphi::detail::CacheRuleHandler cacheRuleset;
    for (std::string &stableState : interpreter.getCacheStableStateNames()) {
      for (llvm::StringRef cpu_event : murphi::detail::cpu_events) {
        nlohmann::json stmts =
            interpreter.getMurphiCacheStatements(stableState, cpu_event);
        if (stmts == nullptr)
          continue;
        cacheRuleset.rules.emplace_back(
            murphi::detail::CPUEventRule{stableState, cpu_event.str()});
      }
    }
    data["rules"].emplace_back(std::move(cacheRuleset));
  }

  void assembleNetworkRulesets(nlohmann::json &data) {
    for (std::pair<std::string, std::string> &network :
         interpreter.getNetworks()) {
      std::string netId = network.first;
      std::string netOrder = network.second;
      if (netOrder == "ordered") {
        data["rules"].push_back(detail::OrderedRuleset{netId});
      } else {
        data["rules"].push_back(detail::UnorderedRuleset{netId});
      }
    }
  }

  void assembleStartStateRules(nlohmann::json &data) {
    auto ss = detail::StartState{"PROTOCOL START STATE", {}, {}};

    /// add the directory and cache start states
    ss.statements.emplace_back(interpreter.getCacheStartState());
    ss.statements.emplace_back(interpreter.getDirectoryStartState());

    /// ** Undefine networks //
    for (std::pair<std::string, std::string> &nw : interpreter.getNetworks()) {
      auto netId = nw.first;
      auto ordering = nw.second;
      auto rhs = detail::Designator{netId, {}};
      ss.statements.emplace_back(detail::UndefineStmt<decltype(rhs)>{rhs});
      if (ordering == "ordered")
        ss.statements.emplace_back(
            boilerplate::getOrderedCountStartState(netId));
    }

    data["rules"].push_back(ss);
  }

  void assembleInvariant(nlohmann::json &data) {
    data["rules"].emplace_back(murphi::detail::SWMRInvariant{});
  }

  InterpreterT interpreter;
};

} // namespace murphi