#pragma once
#include "PCC/PCCOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include <inja/inja.hpp>

namespace murphi{
using namespace inja;
using jon = nlohmann::json;

class MurphiTemplateCodeGen{
public:
  MurphiTemplateCodeGen(mlir::ModuleOp op): theModule{op}{}

  void translate();

private:

  mlir::ModuleOp theModule;
  json data;
};

}