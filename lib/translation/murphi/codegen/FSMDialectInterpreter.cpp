#include "translation/murphi/codegen/FSMDialectInterpreter.h"
#include "FSM/FSMOps.h"
#include "FSM/FSMUtils.h"
#include "translation/murphi/codegen/FSMOperationConverter.h"
#include <set>

using namespace mlir;
using namespace mlir::fsm;
using namespace nlohmann;
constexpr size_t default_reserve_amount = 10;
namespace murphi {

json FSMDialectInterpreter::getMurphiCacheStatements(llvm::StringRef state,
                                                     llvm::StringRef action) {
  auto transOp = theModule.lookupSymbol<MachineOp>("cache")
                     .lookupSymbol<StateOp>(state)
                     .lookupSymbol<TransitionOp>(action);
  if (!transOp)
    return nullptr;

  FSMOperationConverter opConverter;

  return opConverter.convert(transOp);
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

} // namespace murphi
