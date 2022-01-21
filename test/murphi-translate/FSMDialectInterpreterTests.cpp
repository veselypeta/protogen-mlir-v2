#include "../FSM_Dialect/OpHelper.h"
#include "../FSM_Dialect/fixtures.h"
#include "mlir/Parser.h"
#include "translation/murphi/codegen/FSMDialectInterpreter.h"
#include <gtest/gtest.h>

using namespace mlir;
using namespace mlir::fsm;
using namespace murphi;

TEST(FSMDialectInterpreter, getCacheStates) {
  // setup
  OpHelper helper;
  auto result = parseSourceString(mlirMIFull, &helper.ctx);
  ASSERT_NE(*result, nullptr);
  FSMDialectInterpreter interpreter(*result);

  auto cacheStateNames = interpreter.getCacheStateNames();
  EXPECT_EQ(cacheStateNames.size(), 5);
  EXPECT_NE(
      std::find(std::begin(cacheStateNames), std::end(cacheStateNames), "I"),
      std::end(cacheStateNames));
  EXPECT_NE(
      std::find(std::begin(cacheStateNames), std::end(cacheStateNames), "I_load"),
      std::end(cacheStateNames));
  EXPECT_NE(
      std::find(std::begin(cacheStateNames), std::end(cacheStateNames), "M_evict"),
      std::end(cacheStateNames));
}

TEST(FSMDialectInterpreter, getMessageNames){
  // setup
  OpHelper helper;
  auto result = parseSourceString(mlirMIFull, &helper.ctx);
  ASSERT_NE(*result, nullptr);
  FSMDialectInterpreter interpreter(*result);

  auto messageNames = interpreter.getMessageNames();

  EXPECT_EQ(messageNames.size(), 7);

  EXPECT_NE(
      std::find(std::begin(messageNames), std::end(messageNames), "Fwd_GetM"),
      std::end(messageNames));

  EXPECT_NE(
      std::find(std::begin(messageNames), std::end(messageNames), "PutM"),
      std::end(messageNames));

}