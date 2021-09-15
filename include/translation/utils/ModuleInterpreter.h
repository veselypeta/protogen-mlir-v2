#pragma once
#include "mlir/IR/BuiltinOps.h"
#include "PCC/PCCOps.h"

class ModuleInterpreter {
public:
  explicit ModuleInterpreter(mlir::ModuleOp op) : theModule{op} {}

  template <class T> std::vector<T> getOperations();

  std::vector<mlir::pcc::ConstantOp> getConstants();

  std::vector<mlir::pcc::NetDeclOp> getNetworks();

  mlir::pcc::CacheDeclOp getCache();

  mlir::pcc::DirectoryDeclOp getDirectory();

  // FIXME - stub implementation
  static std::vector<std::string> getEnumMessageTypes();

  // FIXME - stub implementation
  static std::vector<std::string> getEnumMachineStates(const std::string &mach);

private:
  mlir::ModuleOp theModule;
  mlir::Block &getModuleBody() {
    return theModule.getOperation()->getRegion(0).front();
  }
};