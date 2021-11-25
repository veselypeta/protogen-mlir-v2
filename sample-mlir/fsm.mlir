// INPUT
pcc.process @cache_M_evict(%arg0: !pcc.struct<!pcc.state<M>, !pcc.data>) {
    %10 = pcc.struct_access %arg0["ID"] : !pcc.struct<!pcc.state<I>, !pcc.data> -> !pcc.id
    %11 = pcc.struct_access %5["ID"] : !pcc.struct<!pcc.state<I>, !pcc.data, !pcc.id> -> !pcc.id
    %12 = pcc.msg_constr %10 %11 {mtype = !pcc.mtype<PutM>} : !pcc.struct<!pcc.id, !pcc.mtype<none>, !pcc.id, !pcc.data>
    pcc.send : %3 : !pcc.network<unordered> %12 : !pcc.struct<!pcc.id, !pcc.mtype<none>, !pcc.id, !pcc.data>
    pcc.await :  {
      pcc.when [ Put_Ack ] (%arg1: !pcc.struct<!pcc.id, !pcc.mtype<none>, !pcc.id>) {
        %14 = pcc.inl_const #pcc.state_attr<I>
        pcc.update [State] %arg0:!pcc.struct<!pcc.state<I>, !pcc.data> %14:pcc.State<I>
        pcc.break
      }
    }
}


// OUTPUT
pcc.process @cache_M_evict(%arg0: !pcc.struct<!pcc.state<M>, !pcc.data>) {
    %10 = pcc.struct_access %arg0["ID"] : !pcc.struct<!pcc.state<I>, !pcc.data> -> !pcc.id
    %11 = pcc.struct_access %5["ID"] : !pcc.struct<!pcc.state<I>, !pcc.data, !pcc.id> -> !pcc.id
    %12 = pcc.msg_constr %10 %11 {mtype = !pcc.mtype<PutM>} : !pcc.struct<!pcc.id, !pcc.mtype<none>, !pcc.id, !pcc.data>
    pcc.send : %3 : !pcc.network<unordered> %12 : !pcc.struct<!pcc.id, !pcc.mtype<none>, !pcc.id, !pcc.data>
    %14 = pcc.inl_const #pcc.state_attr<M_evict>
    pcc.update [State] %arg0:!pcc.struct<!pcc.state<M>, !pcc.data> %14:pcc.State<M_evict>
}

pcc.process @cache_M_evict_Put_Ack(%arg0: !pcc.struct<!pcc.state<M_evict>, !pcc.data>) {
    %14 = pcc.inl_const #pcc.state_attr<I>
    pcc.update [State] %arg0:!pcc.struct<!pcc.state<M_evict>, !pcc.data> %14:pcc.State<I>
}

