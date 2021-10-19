#include <gtest/gtest.h>
#include "mlir-gen/mlirGen.h"
#include "PCC/PCCOps.h"

using namespace mlir;
using namespace mlir::pcc;
TEST(QuickTest, BasicTest){
    ASSERT_EQ(1+3, 4);
}


TEST(StateAttribute, CastingAwayType){
  MLIRContext ctx;
  ctx.getOrLoadDialect<mlir::pcc::PCCDialect>();
  StateAttr stateAttr = StateAttr::get(StateType::get(&ctx, "M"));

  ASSERT_TRUE(stateAttr.getType().isa<StateType>());
  Attribute castedStateAttr = static_cast<Attribute>(stateAttr);
  ASSERT_TRUE(castedStateAttr.getType().isa<StateType>());
}


