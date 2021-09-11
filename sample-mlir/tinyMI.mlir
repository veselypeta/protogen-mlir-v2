module {
      %0 = pcc.constant {id = "NrCaches", val = 3 : i64} : i64
      %1 = pcc.net_decl {netId = "fwd"} : !pcc.network<ordered>
      %2 = pcc.net_decl {netId = "resp"} : !pcc.network<unordered>
      %3 = pcc.net_decl {netId = "req"} : !pcc.network<unordered>
}