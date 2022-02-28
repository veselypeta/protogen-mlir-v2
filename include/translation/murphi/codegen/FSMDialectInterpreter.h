#pragma once
#include "translation/murphi/codegen/MurphiStructs.h"
#include "mlir/IR/BuiltinOps.h"
#include "FSM/FSMOps.h"
#include "nlohmann/json.hpp"
#include "translation/utils/utils.h"
#include <set>
#include <vector>

namespace murphi {

/// Used for hashing Set Types
class TypeHash{
public:
  size_t operator()(const mlir::fsm::SetType &s) const {
    return hash_value(s);
  }
};

class FSMDialectInterpreter {
public:
  explicit FSMDialectInterpreter(mlir::ModuleOp op);
  /// We can use StringRef since the lifetime of this class
  /// will be the same as the MurphiAssembler

  /// Get the names of every message sent in the protocol
  /// i.e. GetM, Fwd_GetM, Put_Ack etc...
  std::vector<std::string> getMessageNames();

  /// Returns the global message type shared by all messages
  nlohmann::json getMessageType();

  /// Returns the names of the types of messasges that can be sent
  /// in the protocol i.e. Resp or Ack
  std::vector<std::string> getMessageTypeNames();

  nlohmann::json getMessageFactory(std::string &msgType);

  /// Get the names of every state in the cache
  /// i.e. I, I_load, M_evict etc...
  std::vector<std::string> getCacheStateNames();

  /// Get the names of every state in the directory
  /// i.e. I, M, O etc...
  std::vector<std::string> getDirectoryStateNames();

  /// get the number of caches in the simulation
  constexpr size_t getCacheSetSize() { return 3; };

  /// return the type decls for any required sets used in the protocol
  /// Set naming is the following:
  /// v_{{ size_of_set}}_{{index_type}}
  /// i.e. v_3_Machines
  nlohmann::json getSetTypes();

  /// return the implementation of each set operations for every set type
  /// i.e. add, contains, del, etc.
  nlohmann::json getSetOperationImpl();

  nlohmann::json getCacheType();
  nlohmann::json getDirectoryType();

  /// Returns a vector of pairs of the form
  /// {"fwd", "ordered"} etc..
  std::vector<std::pair<std::string, std::string>> getNetworks();

  /// returns a JSON array of MurphiStatement
  nlohmann::json getMurphiCacheStatements(llvm::StringRef state,
                                          llvm::StringRef action);
  nlohmann::json getMurphiDirectoryStatements(llvm::StringRef state,
                                              llvm::StringRef action);

  /// returns a vector of string which are the names of each stable state in the
  /// cache
  std::vector<std::string> getCacheStableStateNames();

  /// returns the json statements to implement the rules for the cache/directory's start
  /// state
  nlohmann::json getCacheStartState();
  nlohmann::json getDirectoryStartState();

  bool isMulticastEnabled();

private:
  mlir::ModuleOp theModule;
  std::set<std::pair<std::string, std::string>> mTypeElems;
  std::unordered_map<mlir::fsm::SetType, murphi::detail::Set, TypeHash> setTypeMap;
};

} // namespace murphi
