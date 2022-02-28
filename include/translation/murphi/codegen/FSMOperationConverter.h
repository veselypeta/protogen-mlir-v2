#pragma once
#include "FSM/FSMOps.h"
#include "nlohmann/json.hpp"
#include "llvm/ADT/ScopedHashTable.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
namespace murphi {

/// This class is responsible for converting operations
/// To Murphi json representations
class FSMOperationConverter {
public:
  FSMOperationConverter()=default;
  FSMOperationConverter(const FSMOperationConverter &) = delete;
  FSMOperationConverter &operator=(const FSMOperationConverter &) = delete;
  FSMOperationConverter(FSMOperationConverter &&) = delete;
  FSMOperationConverter &operator=(FSMOperationConverter &&) = delete;

  nlohmann::json convert(mlir::fsm::TransitionOp op);

private:
  nlohmann::json convert(mlir::fsm::MessageOp msgOp);
  nlohmann::json convert(mlir::fsm::UpdateOp op);
  void convert(mlir::fsm::ReferenceOp op);
  void convert(mlir::fsm::AccessOp op);
  void convert(mlir::ConstantOp op);
  void convert(mlir::fsm::ConstOp op);
  void convert(mlir::fsm::CompareOp op);
  void convert(mlir::fsm::AddOp op);
  void convert(mlir::fsm::SetContains op);
  void convert(mlir::fsm::SetCount op);
  nlohmann::json convert(mlir::fsm::SetAdd op);
  nlohmann::json convert(mlir::fsm::SetDelete op);
  nlohmann::json convert(mlir::fsm::SetClear op);
  nlohmann::json convert(mlir::fsm::SendOp op);
  nlohmann::json convert(mlir::fsm::IfOp op);
  nlohmann::json convert(mlir::fsm::MulticastOp op);
  nlohmann::json convert(mlir::fsm::CallOp op);
  nlohmann::json convert(mlir::Operation *op);
  void setupSymbolTable(mlir::fsm::TransitionOp op);

  llvm::ScopedHashTable<mlir::Value, std::string> symbolTable;
  using SymbolTableScopeT =
      llvm::ScopedHashTableScope<mlir::Value, std::string>;
};


std::string FSMConvertType(mlir::Type type);

} // namespace murphi