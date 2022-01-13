#include "PassDetail.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace mlir;
using namespace mlir::pcc;
namespace {

class ProtogenRewriter : public PatternRewriter {
public:
  explicit ProtogenRewriter(MLIRContext *ctx) : PatternRewriter(ctx) {}
  // TODO - override necessary methods
};

class StallingProtogenOptimizationPass
    : public StallingProtogenOptimizationPassBase<
          StallingProtogenOptimizationPass> {
public:
  void runOnOperation() override;
};

void StallingProtogenOptimizationPass::runOnOperation() {

  ModuleOp theModule = OperationPass<ModuleOp>::getOperation();

  auto result = theModule.walk([&](ProcessOp processOp) {
    // Identify all transient states
    llvm::StringRef stateType =
        processOp->getAttr("state_type").cast<mlir::StringAttr>().getValue();
    if (stateType != "transient")
      return WalkResult::advance();

    // What messages might be interleaved?
    // messages that can be handled by the logical start state must be handled
    // in the transient state

    auto find_all_start_state_processes = [](ModuleOp &moduleOp,
                                             ProcessOp &queryProcess) {
      auto procOpsIt = moduleOp.getOps<ProcessOp>();
      std::for_each(
          std::begin(procOpsIt), std::end(procOpsIt), [&](ProcessOp p) {
            llvm::StringRef stateType =
                p->getAttr("state_type").cast<StringAttr>().getValue();
            llvm::StringRef startState =
                p->getAttr("start_state").cast<StringAttr>().getValue();
            llvm::StringRef queryLogSS =
                queryProcess->getAttr("logical_start_state")
                    .cast<StringAttr>()
                    .getValue();

            // skip transient sates & states which do not match the query
            // process logical_start_state
            if (stateType != "transient" && startState == queryLogSS) {
              // Then this is a possible racing message, and we must handle it

              // What do we do???
              // We need to act as if we are in this state i.e. copy the
              // operations We need to look at the directory and see what state
              // it 'sees' us in now and set that as the new logical start state
              ProtogenRewriter rewriter(moduleOp.getContext());

              // we need a name for this new transient state
              // we use the naming scheme trans_state + racing_action
              // i.e. for state M_evict with a Fwd_GetM racing =>
              // M_evict_Fwd_GetM
              llvm::StringRef queryProcName = queryProcess.sym_name();
              llvm::StringRef queryAction =
                  queryProcess->getAttr("action").cast<StringAttr>().getValue();
              llvm::Twine raceTransName = queryProcName + "_" + queryAction;

              // the racing process will have the same type as the query
              // process, since we will be handling it in the same state
              auto raceProcType = queryProcess.getType();

              // next we need to modify the attributes correctly so that the new
              // process resembles the correct state of the FSM

              rewriter.setInsertionPointAfter(queryProcess);

              ProcessOp concurrentHandler __attribute__((unused)) = rewriter.create<ProcessOp>(
                  queryProcess.getLoc(), raceTransName.str(), raceProcType);
            }
          });
    };

    find_all_start_state_processes(theModule, processOp);

    return WalkResult::advance();
  });

  if (result.wasInterrupted())
    return signalPassFailure();
}

} // namespace

namespace mlir {
namespace pcc {

std::unique_ptr<OperationPass<ModuleOp>>
createStallingProtogenOptimizationPass() {
  return std::make_unique<StallingProtogenOptimizationPass>();
}

} // namespace pcc
} // namespace mlir