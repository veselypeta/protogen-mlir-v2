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

  void assembleTypes(nlohmann::json &data) {
    assembleEnums(data);
    // Message Type
    data["decls"]["type_decls"].emplace_back(interpreter.getMessageType());
    assembleMachineEntryTypes(data);
  }
  void assembleDecls(nlohmann::json &data) {
    assembleConstants(data);
    assembleTypes(data);
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
    assembleCacheController(data);
    assembleDirectoryController(data);
    assembleStartStateFunctions(data);
  }

  void assembleRules(nlohmann::json &data) {
    assembleCacheRuleset(data);
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

  void assembleInvariant(nlohmann::json &data) {
    data["rules"].emplace_back(murphi::detail::SWMRInvariant{});
  }

  InterpreterT interpreter;
};

} // namespace murphi