#pragma once
#include "FSM/FSMOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

// Helper class to construct ops
class OpHelper {
public:
  OpHelper() : builder{&ctx} {
    ctx.getOrLoadDialect<mlir::fsm::FSMDialect>();
    ctx.getOrLoadDialect<mlir::StandardOpsDialect>();
    module = mlir::ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToStart(module.getBody());
  }

  mlir::MLIRContext ctx;
  mlir::OpBuilder builder;
  mlir::ModuleOp module;
};