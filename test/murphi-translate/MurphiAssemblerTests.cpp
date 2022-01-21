#include "../FSM_Dialect/OpHelper.h"
#include "../FSM_Dialect/fixtures.h"
#include "mlir/Parser.h"
#include "translation/murphi/codegen/FSMDialectInterpreter.h"
#include "translation/murphi/codegen/MurphiCodeGen.h"
#include <gtest/gtest.h>

using namespace murphi;
TEST(MurphiAssembler, dd){
  // setup
  OpHelper helper;
  auto result = parseSourceString(mlirMIFull, &helper.ctx);
  ASSERT_NE(*result, nullptr);
  FSMDialectInterpreter interpreter(*result);
  MurphiAssembler<FSMDialectInterpreter> assembler{interpreter};

  auto assembly = assembler.assemble();
}