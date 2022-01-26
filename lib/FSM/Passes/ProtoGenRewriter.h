#pragma once
#include "mlir/IR/PatternMatch.h"

// We need a custom rewriter to be able to construct it
class ProtoGenRewriter : public mlir::PatternRewriter {
public:
  explicit ProtoGenRewriter(mlir::MLIRContext *ctx) : PatternRewriter{ctx} {
    // TODO - overwrite necessary methods here
  }
};