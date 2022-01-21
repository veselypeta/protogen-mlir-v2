#include "translation/murphi/codegen/FSMDialectInterpreter.h"
#include "FSM/FSMOps.h"
#include "FSM/FSMUtils.h"

using namespace mlir;
using namespace mlir::fsm;
constexpr size_t default_reserve_amount = 10;
namespace murphi {

std::vector<std::string> FSMDialectInterpreter::getMessageNames() {
  std::vector<std::string> messageNames;
  messageNames.reserve(default_reserve_amount);

  std::vector<MessageOp> allSentMessages;
  allSentMessages.reserve(default_reserve_amount);

  utils::searchFor<MessageOp>(theModule.getOperation(), allSentMessages);

  for (auto msgOp : allSentMessages) {
    messageNames.push_back(msgOp.msgName().str());
  }
  return messageNames;
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
