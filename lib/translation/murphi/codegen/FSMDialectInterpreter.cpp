#include "translation/murphi/codegen/FSMDialectInterpreter.h"
#include "FSM/FSMOps.h"
#include "FSM/FSMUtils.h"
#include "translation/murphi/codegen/FSMOperationConverter.h"
#include <set>

using namespace mlir;
using namespace mlir::fsm;
using namespace nlohmann;
constexpr size_t default_reserve_amount = 10;

namespace {

json getMurphiMachineStatements(llvm::StringRef state, llvm::StringRef action,
                                llvm::StringRef machId, ModuleOp theModule) {
  auto transOp = theModule.lookupSymbol<MachineOp>(machId)
                     .lookupSymbol<StateOp>(state)
                     .lookupSymbol<TransitionOp>(action);
  if (!transOp)
    return nullptr;

  murphi::FSMOperationConverter opConverter;

  return opConverter.convert(transOp);
}
} // namespace
namespace murphi {

json FSMDialectInterpreter::getMurphiCacheStatements(llvm::StringRef state,
                                                     llvm::StringRef action) {
  return getMurphiMachineStatements(state, action, "cache", theModule);
}

json FSMDialectInterpreter::getMurphiDirectoryStatements(
    llvm::StringRef state, llvm::StringRef action) {
  return getMurphiMachineStatements(state, action, "directory", theModule);
}

std::vector<std::string> FSMDialectInterpreter::getMessageNames() {
  std::set<std::string> messageNames;

  std::vector<MessageOp> allSentMessages;
  allSentMessages.reserve(default_reserve_amount);

  utils::searchFor<MessageOp>(theModule.getOperation(), allSentMessages);

  for (auto msgOp : allSentMessages) {
    messageNames.insert(msgOp.msgName().str());
  }
  return {std::begin(messageNames), std::end(messageNames)};
}

std::vector<std::string> FSMDialectInterpreter::getCacheStateNames() {
  std::vector<std::string> stateNames;
  stateNames.reserve(default_reserve_amount);

  std::vector<StateOp> allCacheStateOps;
  allCacheStateOps.reserve(default_reserve_amount);
  Operation *theCache = theModule.lookupSymbol("cache");
  utils::searchFor<StateOp>(theCache, allCacheStateOps);

  for (auto stateOp : allCacheStateOps) {
    stateNames.push_back(stateOp.sym_name().str());
  }

  return stateNames;
}

std::vector<std::string> FSMDialectInterpreter::getDirectoryStateNames() {
  std::vector<std::string> stateNames;
  stateNames.reserve(default_reserve_amount);

  std::vector<StateOp> allDirectoryStates;
  allDirectoryStates.reserve(default_reserve_amount);
  Operation *theCache = theModule.lookupSymbol("directory");
  utils::searchFor<StateOp>(theCache, allDirectoryStates);

  for (auto stateOp : allDirectoryStates) {
    stateNames.push_back(stateOp.sym_name().str());
  }

  return stateNames;
}

std::vector<std::string> FSMDialectInterpreter::getCacheStableStateNames() {
  std::vector<StateOp> stableStates;
  utils::searchForIf(theModule, stableStates, [](StateOp stateOp) {
    return !utils::isTransientState(stateOp);
  });

  std::set<std::string> outs;
  std::for_each(std::begin(stableStates), std::end(stableStates),
                [&outs](StateOp op) { outs.insert(op.sym_name().str()); });
  return {outs.begin(), outs.end()};
}

} // namespace murphi
