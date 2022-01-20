#include "FSM/FSMOps.h"
#include "FSM/FSMUtils.h"
#include "OpHelper.h"
#include "mlir/Parser.h"
#include "mlir/IR/DialectImplementation.h"
#include <gtest/gtest.h>

using namespace mlir;
using namespace mlir::fsm;
using namespace mlir::fsm::utils;

namespace {
llvm::StringRef mlirMIFull =
    "module {\n"
    "\n"
    "fsm.machine @cache(){\n"
    "    %State = fsm.variable \"State\" {initValue = \"I\"} : !fsm.state\n"
    "    %cl = fsm.variable \"cl\" : !fsm.data\n"
    "\n"
    "    fsm.state @I transitions {\n"
    "        fsm.transition @store() attributes {nextState=@I_store} {\n"
    "            %src = fsm.ref @cache\n"
    "            %dst = fsm.ref @directory\n"
    "            %msg = fsm.message @Request \"GetM\" %src, %dst : !fsm.id, "
    "!fsm.id -> !fsm.msg\n"
    "        }\n"
    "\n"
    "        fsm.transition @load() attributes {nextState=@I_load} {\n"
    "            %src = fsm.ref @cache\n"
    "            %dst = fsm.ref @directory\n"
    "            %msg = fsm.message @Request \"GetM\" %src, %dst : !fsm.id, "
    "!fsm.id -> !fsm.msg\n"
    "        }\n"
    "    }\n"
    "\n"
    "    fsm.state @I_store {prevTransition=@I::@store} transitions {\n"
    "\n"
    "        fsm.transition @GetM_Ack_D(%msg : !fsm.msg) attributes {nextState "
    "= @M}{\n"
    "            %msg_cl = fsm.access {memberId = \"cl\"} %msg : !fsm.msg -> "
    "!fsm.data\n"
    "            fsm.update %cl, %msg_cl : !fsm.data\n"
    "        }\n"
    "\n"
    "    }\n"
    "\n"
    "    fsm.state @I_load {prevTransition=@I::@load} transitions {\n"
    "        fsm.transition @GetM_Ack_D(%msg : !fsm.msg) attributes {nextState "
    "= @M}{\n"
    "            %msg_cl = fsm.access {memberId = \"cl\"} %msg : !fsm.msg -> "
    "!fsm.data\n"
    "            fsm.update %cl, %msg_cl : !fsm.data\n"
    "        }\n"
    "    }\n"
    "\n"
    "    fsm.state @M transitions {\n"
    "        fsm.transition @load(){\n"
    "            fsm.nop\n"
    "        }\n"
    "\n"
    "        fsm.transition @store(){\n"
    "            fsm.nop\n"
    "        }\n"
    "\n"
    "        fsm.transition @Fwd_GetM(%Fwd_GetM : !fsm.msg) attributes "
    "{nextState = @I}{\n"
    "            %src = fsm.ref @cache\n"
    "            %dst = fsm.access {memberId = \"src\"} %Fwd_GetM : !fsm.msg "
    "-> !fsm.id\n"
    "            %sent_msg = fsm.message @Resp \"GetM_Ack_D\" %src, %dst, %cl "
    ": !fsm.id, !fsm.id, !fsm.data -> !fsm.msg\n"
    "        }\n"
    "\n"
    "        fsm.transition @evict() attributes {nextState = @M_evict}{\n"
    "            %src = fsm.ref @cache\n"
    "            %dst = fsm.ref @directory\n"
    "            %msg = fsm.message @Resp \"PutM\" %src, %dst, %cl : !fsm.id, "
    "!fsm.id, !fsm.data -> !fsm.msg\n"
    "        }\n"
    "    }\n"
    "\n"
    "    fsm.state @M_evict {prevTransition = @M::@evict} transitions {\n"
    "        fsm.transition @Put_Ack(%arg0 : !fsm.msg) attributes {nextState = "
    "@I}{\n"
    "            fsm.nop\n"
    "        }\n"
    "    }\n"
    "\n"
    "}\n"
    "\n"
    "fsm.machine @directory(){\n"
    "    %State = fsm.variable \"State\" {initValue = \"I\"} : !fsm.state\n"
    "    %cl = fsm.variable \"cl\" : !fsm.data\n"
    "    %owner = fsm.variable \"owner\" : !fsm.id\n"
    "\n"
    "    fsm.state @I transitions {\n"
    "        fsm.transition @GetM(%GetM : !fsm.msg) attributes {nextState = "
    "@M}{\n"
    "            %src = fsm.ref @directory\n"
    "            %dst = fsm.access {memberId = \"src\"} %GetM : !fsm.msg -> "
    "!fsm.id\n"
    "            %msg = fsm.message @Resp \"GetM_Ack_D\" %src, %dst, %cl : "
    "!fsm.id, !fsm.id, !fsm.data -> !fsm.msg\n"
    "            fsm.update %owner, %dst : !fsm.id\n"
    "        }\n"
    "    }\n"
    "\n"
    "    fsm.state @M transitions {\n"
    "        fsm.transition @GetM(%GetM : !fsm.msg) attributes {nextState = "
    "@M}{\n"
    "            %src = fsm.access {memberId = \"src\"} %GetM : !fsm.msg -> "
    "!fsm.id\n"
    "            %msg = fsm.message @Request \"Fwd_GetM\" %src, %owner : "
    "!fsm.id, !fsm.id -> !fsm.msg\n"
    "        }\n"
    "\n"
    "        fsm.transition @PutM(%PutM : !fsm.msg) {\n"
    "            %dst = fsm.access {memberId = \"src\"} %PutM : !fsm.msg -> "
    "!fsm.id\n"
    "            %src = fsm.ref @directory\n"
    "            %msg = fsm.message @Ack \"Put_Ack\" %src, %dst : !fsm.id, "
    "!fsm.id -> !fsm.msg\n"
    "            %true = constant true\n"
    "            fsm.if %true {\n"
    "                %n_cl = fsm.access {memberId = \"cl\"} %PutM : !fsm.msg "
    "-> !fsm.data\n"
    "                fsm.update %cl, %n_cl : !fsm.data\n"
    "                // TODO - figure out how to update state\n"
    "            }\n"
    "\n"
    "        }\n"
    "    }\n"
    "}\n"
    "\n"
    "}";
}

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


