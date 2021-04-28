#ifndef PCC_OPS
#define PCC_OPS

def FooOp : PCC_Op<"foo"> {
    let summary = "foo op";
    let description = [{}];
    let results = (outs PCCType:$results);
    let assemblyFormat = [{
        attr-dict `:` type($results)
    }];
}

#endif // PCC_OPS