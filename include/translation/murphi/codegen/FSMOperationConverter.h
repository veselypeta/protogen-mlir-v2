#pragma once
#include "FSM/FSMOps.h"
#include "nlohmann/json.hpp"
#include "llvm/ADT/ScopedHashTable.h"
namespace murphi {

/// This class is responsible for converting operations
/// To Murphi json representations
class FSMOperationConverter {
public:
  FSMOperationConverter() {
    // setup the symbol table to correctly
    SymbolTableScopeT global_scope(symbolTable);
  }
  FSMOperationConverter(const FSMOperationConverter &) = delete;
  FSMOperationConverter &operator=(const FSMOperationConverter &) = delete;
  FSMOperationConverter(FSMOperationConverter &&) = delete;
  FSMOperationConverter &operator=(FSMOperationConverter &&) = delete;

  nlohmann::json convert(mlir::fsm::TransitionOp op);

private:
  nlohmann::json convert(mlir::fsm::MessageOp msgOp);
  nlohmann::json convert(mlir::Operation *op);
  void setupSymbolTable(mlir::fsm::TransitionOp op);

  llvm::ScopedHashTable<mlir::Value, std::string> symbolTable;
  using SymbolTableScopeT =
      llvm::ScopedHashTableScope<mlir::Value, std::string>;
};

} // namespace murphi