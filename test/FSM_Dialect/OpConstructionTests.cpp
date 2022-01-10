//
// Created by petr on 10/01/2022.
//
#include "FSM/FSMOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Parser.h"
#include <gtest/gtest.h>

using namespace mlir;
using namespace mlir::fsm;

// Helper class to construct ops
class OpHelper {
public:
  OpHelper() : builder{&ctx} {
    ctx.getOrLoadDialect<FSMDialect>();
    ctx.getOrLoadDialect<StandardOpsDialect>();
    module = ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToStart(module.getBody());
  }

  MLIRContext ctx;
  OpBuilder builder;
  ModuleOp module;
};

TEST(OpConstructionTests, PrintMachineOp) {

  OpHelper helper;

  FunctionType type = helper.builder.getFunctionType(llvm::None, llvm::None);

  auto machineOp = helper.builder.create<MachineOp>(
      helper.builder.getUnknownLoc(), "cache", type);
  Block *entry = machineOp.addEntryBlock();

  //// add a constant to the body
  // set the insertion point to inside the body of the machine
  helper.builder.setInsertionPointToStart(entry);

  // create the constant op
  auto initAttr = helper.builder.getI64IntegerAttr(21);
  helper.builder.create<VariableOp>(helper.builder.getUnknownLoc(),
                                    initAttr.getType(), initAttr, "some_count");

  // print operation to string
  std::string str;
  llvm::raw_string_ostream stream{str};
  machineOp.print(stream);

  ASSERT_STREQ(str.c_str(),
               "fsm.machine @cache() {\n  %some_count = fsm.variable "
               "\"some_count\" {initValue = 21 : i64} : i64\n}");
}

TEST(OpConstructionTests, ParseMachineOp) {
  OpHelper helper;
  llvm::StringRef machineOp ="fsm.machine @cache() {\n  %some_count = fsm.variable "
                              "\"some_count\" {initValue = 21 : i64} : i64\n}";
  auto result = parseSourceString<ModuleOp>(machineOp, &helper.ctx);
  result->walk([&](MachineOp op) {
    ASSERT_STREQ(op.sym_name().str().c_str(), "cache");
  });
  result->walk([&](VariableOp op) {
    ASSERT_EQ(op.initValue().cast<IntegerAttr>().getInt(), 21);
  });
}
