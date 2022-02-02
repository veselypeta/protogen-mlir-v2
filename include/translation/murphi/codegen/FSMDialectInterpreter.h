#pragma once
#include "mlir/IR/BuiltinOps.h"
#include "nlohmann/json.hpp"
#include <vector>
#include <set>
#include "translation/utils/utils.h"

namespace murphi {

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

private:
  mlir::ModuleOp theModule;
  std::set<std::pair<std::string, std::string>> mTypeElems;
};

} // namespace murphi
