#include "FSM/FSMOps.h"
#include "FSM/FSMUtils.h"
#include "OpHelper.h"
#include "mlir/Parser.h"
#include "mlir/IR/DialectImplementation.h"
#include <gtest/gtest.h>
#include "fixtures.h"

using namespace mlir;
using namespace mlir::fsm;
using namespace mlir::fsm::utils;


TEST(FSMUtils, MsgMatcher) {
  OpHelper helper;
  Location unknLoc = helper.builder.getUnknownLoc();

  FunctionType cacheFnType = helper.builder.getFunctionType({}, {});
  MachineOp theCache =
      helper.builder.create<MachineOp>(unknLoc, "cache", cacheFnType);
  Block *cacheEntry = theCache.addEntryBlock();
  helper.builder.setInsertionPointToStart(cacheEntry);

  // create I state
  StateOp iState = helper.builder.create<StateOp>(unknLoc, "I");
  Block *iStateEntry = iState.addEntryBlock();
  helper.builder.setInsertionPointToStart(iStateEntry);

  // i - load trans
  FunctionType iLoadFunType = helper.builder.getFunctionType({}, {});
  TransitionOp iLoadTran = helper.builder.create<TransitionOp>(
      unknLoc, "load", iLoadFunType, nullptr);
  Block *iLoadEntry = iLoadTran.addEntryBlock();
  helper.builder.setInsertionPointToStart(iLoadEntry);

  // send msg
  helper.builder.create<MessageOp>(unknLoc, MsgType::get(&helper.ctx),
                                   helper.builder.getSymbolRefAttr("Resp"),
                                   helper.builder.getStringAttr("GetM"),
                                   llvm::None);

  std::vector<MessageOp> allMsgs;
  searchFor<MessageOp>(helper.module.getOperation(), allMsgs);
  EXPECT_EQ(allMsgs.size(), 1);

  MessageOp foundMsg = allMsgs.at(0);
  EXPECT_EQ(foundMsg.msgName(), "GetM");
  EXPECT_EQ(foundMsg.msgType().getLeafReference(), "Resp");
}

TEST(UtilsTests, ParseMI){
  OpHelper helper;
  auto result = parseSourceString(mlirMIFull, &helper.ctx);
  ASSERT_NE(*result, nullptr);
}

TEST(UtilsTest, SearchForFn){
  OpHelper helper;
  auto result = parseSourceString(mlirMIFull, &helper.ctx);
  ASSERT_NE(*result, nullptr);

  std::vector<VariableOp> vars;
  searchFor(*result, vars);

  EXPECT_EQ(vars.size(), 5);
}

TEST(UtilsTest, SearchForIfFn){
  OpHelper helper;
  auto result = parseSourceString(mlirMIFull, &helper.ctx);
  ASSERT_NE(*result, nullptr);

  std::vector<VariableOp> vars;
  searchForIf(*result, vars, [](VariableOp var){ return !var.initValue().hasValue(); });

  EXPECT_EQ(vars.size(), 3);
}


