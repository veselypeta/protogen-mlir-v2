#pragma once
#include "mlir/IR/BuiltinOps.h"
#include <vector>

namespace murphi {

class FSMDialectInterpreter {
public:
  explicit FSMDialectInterpreter(mlir::ModuleOp op) : theModule{op} {}
  /// We can use StringRef since the lifetime of this class
  /// will be the same as the MurphiAssembler

  /// Get the names of every message sent in the protocol
  /// i.e. GetM, Fwd_GetM, Put_Ack etc...
  std::vector<std::string> getMessageNames();

  /// Get the names of every state in the cache
  /// i.e. I, I_load, M_evict etc...
  std::vector<std::string> getCacheStateNames();

  /// Get the names of every state in the directory
  /// i.e. I, M, O etc...
  std::vector<std::string> getDirectoryStateNames();

  /// get the number of caches in the simulation
  constexpr size_t getCacheSetSize() { return 3; };

private:
  mlir::ModuleOp theModule;
};

} // namespace murphi
