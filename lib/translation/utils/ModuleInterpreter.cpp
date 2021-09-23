#include "translation/utils/ModuleInterpreter.h"
#include <algorithm>

template <class T> std::vector<T> ModuleInterpreter::getOperations() {
  std::vector<T> ops;
  for (auto &op : getModuleBody()) {
    if (mlir::isa<T>(op)) {
      ops.push_back(mlir::dyn_cast<T>(op));
    }
  }
  return ops;
}

std::vector<mlir::pcc::ConstantOp> ModuleInterpreter::getConstants() {
  return getOperations<mlir::pcc::ConstantOp>();
}

std::vector<mlir::pcc::NetDeclOp> ModuleInterpreter::getNetworks() {
  return getOperations<mlir::pcc::NetDeclOp>();
}

mlir::pcc::CacheDeclOp ModuleInterpreter::getCache() {
  auto cacheOps = getOperations<mlir::pcc::CacheDeclOp>();
  assert(cacheOps.size() == 1 &&
         "found zero or more than one cache decl operations");
  return cacheOps.at(0);
}

mlir::pcc::DirectoryDeclOp ModuleInterpreter::getDirectory() {
  auto directoryOps = getOperations<mlir::pcc::DirectoryDeclOp>();
  assert(directoryOps.size() == 1 &&
         "found zero or more directory decl operations!");
  return directoryOps.at(0);
}

std::vector<std::string> ModuleInterpreter::getEnumMessageTypes() {
  return {"Get_M", "Fwd_Get_M", "GetM_Ack_D"};
}

std::vector<std::string>
ModuleInterpreter::getEnumMachineStates(const std::string &mach) {
  std::vector<std::string> states = {"I", "M", "I_load", "M_evict"};
  std::for_each(states.begin(), states.end(),
                [&mach](auto &state) { state = mach + "_" + state; });
  return states;
}

std::vector<mlir::pcc::MsgDeclOp> ModuleInterpreter::getMessages() {
  return getOperations<mlir::pcc::MsgDeclOp>();
}

mlir::pcc::MsgDeclOp ModuleInterpreter::getMessage(llvm::StringRef constructor) {
  auto allMessageConstructorTypes = getMessages();
  auto result = std::find_if(allMessageConstructorTypes.begin(), allMessageConstructorTypes.end(), [&constructor](mlir::pcc::MsgDeclOp &msgDeclOp){
    return msgDeclOp.id() == constructor;
  });
  assert(result != allMessageConstructorTypes.end() && "Could not find Message constructor with that identifier!");
  return *result;
}
