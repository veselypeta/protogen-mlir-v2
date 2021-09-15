#include <gtest/gtest.h>
#include "translation/utils/ModuleInterpreter.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"


class ModuleInterpreterFixture: public ::testing::Test {
  public :
      ModuleInterpreterFixture(): builder{&ctx} {
        ctx.getOrLoadDialect<mlir::pcc::PCCDialect>();
        ctx.getOrLoadDialect<mlir::StandardOpsDialect>();
        theModule = mlir::ModuleOp::create(builder.getUnknownLoc());
        builder.setInsertionPointToStart(theModule.getBody());
      }

      mlir::MLIRContext ctx;
      mlir::OpBuilder builder;
      mlir::ModuleOp theModule;

};

TEST_F(ModuleInterpreterFixture, getConstants){
  ModuleInterpreter moduleInterpreter{theModule};

  // create a basic constant op
  builder.create<mlir::pcc::ConstantOp>(builder.getUnknownLoc(),"hello", 21);
  builder.create<mlir::pcc::ConstantOp>(builder.getUnknownLoc(),"nrCaches", 4);


  std::vector<mlir::pcc::ConstantOp> constOps = moduleInterpreter.getConstants();

  ASSERT_EQ(constOps.size(), 2);

  auto &firstConst = constOps.at(0);
  ASSERT_STREQ(firstConst.id().str().c_str(), "hello");
  ASSERT_EQ(firstConst.val(), 21);

  auto &secondConst = constOps.at(1);
  ASSERT_STREQ(secondConst.id().str().c_str(), "nrCaches");
  ASSERT_EQ(secondConst.val(), 4);
}