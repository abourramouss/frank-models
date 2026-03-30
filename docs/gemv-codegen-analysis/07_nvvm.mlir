// -----// IR Dump After ConvertToNVVMPass (iree-convert-to-nvvm) //-----
 //
module {
  llvm.func @__nv_fmaf(f32, f32, f32) -> f32
  llvm.func @gemv_f16_dispatch_0_matvec_like_2048x1024_f16(%arg0: !llvm.ptr<1> {llvm.align = 16 : i32, llvm.noalias, llvm.nonnull, llvm.noundef, llvm.readonly}, %arg1: !llvm.ptr<1> {llvm.align = 16 : i32, llvm.noalias, llvm.nonnull, llvm.noundef, llvm.readonly}, %arg2: !llvm.ptr<1> {llvm.align = 16 : i32, llvm.noalias, llvm.nonnull, llvm.noundef}) {
    %0 = llvm.mlir.constant(-1 : i32) : i32
    %1 = llvm.mlir.poison : vector<1xf16>
    %2 = llvm.mlir.constant(7 : i64) : i64
    %3 = llvm.mlir.constant(6 : i64) : i64
    %4 = llvm.mlir.constant(5 : i64) : i64
    %5 = llvm.mlir.constant(4 : i64) : i64
    %6 = llvm.mlir.constant(3 : i64) : i64
    %7 = llvm.mlir.constant(2 : i64) : i64
    %8 = llvm.mlir.constant(1 : i64) : i64
    %9 = llvm.mlir.constant(0 : i64) : i64
    %10 = llvm.mlir.constant(2048 : index) : i64
    %11 = llvm.mlir.constant(64 : index) : i64
    %12 = llvm.mlir.constant(true) : i1
    %13 = llvm.mlir.poison : !llvm.array<1 x array<1 x array<1 x array<1 x array<1 x vector<1xf16>>>>>>
    %14 = llvm.mlir.constant(0.000000e+00 : f16) : f16
    %15 = llvm.mlir.constant(7 : i32) : i32
    %16 = llvm.mlir.constant(6 : i32) : i32
    %17 = llvm.mlir.constant(5 : i32) : i32
    %18 = llvm.mlir.constant(3 : i32) : i32
    %19 = llvm.mlir.constant(128 : i32) : i32
    %20 = llvm.mlir.constant(0 : i32) : i32
    %21 = llvm.mlir.constant(16 : i32) : i32
    %22 = llvm.mlir.constant(8 : i32) : i32
    %23 = llvm.mlir.constant(4 : i32) : i32
    %24 = llvm.mlir.constant(2 : i32) : i32
    %25 = llvm.mlir.constant(32 : i32) : i32
    %26 = llvm.mlir.constant(1 : i32) : i32
    %27 = llvm.mlir.constant(dense<0.000000e+00> : vector<1x1x1x1x1x1xf16>) : !llvm.array<1 x array<1 x array<1 x array<1 x array<1 x vector<1xf16>>>>>>
    %28 = nvvm.read.ptx.sreg.tid.x range <i32, 0, 32> : i32
    %29 = llvm.sext %28 : i32 to i64
    llvm.intr.assume %12 ["align"(%arg0, %11 : !llvm.ptr<1>, i64)] : i1
    llvm.intr.assume %12 ["align"(%arg1, %11 : !llvm.ptr<1>, i64)] : i1
    llvm.intr.assume %12 ["align"(%arg2, %11 : !llvm.ptr<1>, i64)] : i1
    %30 = nvvm.read.ptx.sreg.ctaid.x range <i32, 0, 2048> : i32
    %31 = llvm.sext %30 : i32 to i64
    %32 = llvm.trunc %29 : i64 to i32
    %33 = llvm.mul %32, %22 overflow<nsw> : i32
    llvm.br ^bb1(%20, %27 : i32, !llvm.array<1 x array<1 x array<1 x array<1 x array<1 x vector<1xf16>>>>>>)
  ^bb1(%34: i32, %35: !llvm.array<1 x array<1 x array<1 x array<1 x array<1 x vector<1xf16>>>>>>):  // 2 preds: ^bb0, ^bb2
    %36 = llvm.icmp "slt" %34, %19 : i32
    llvm.cond_br %36, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %37 = llvm.mul %34, %22 overflow<nsw> : i32
    %38 = llvm.add %33, %37 : i32
    %39 = llvm.zext %38 : i32 to i64
    %40 = llvm.getelementptr %arg0[%39] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
    %41 = llvm.load %40 {alignment = 2 : i64} : !llvm.ptr<1> -> vector<8xf16>
    %42 = llvm.mul %39, %10 : i64
    %43 = llvm.add %42, %31 : i64
    %44 = llvm.getelementptr %arg1[%43] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
    %45 = llvm.load %44 {alignment = 2 : i64} : !llvm.ptr<1> -> vector<1xf16>
    %46 = llvm.add %38, %26 overflow<nsw> : i32
    %47 = llvm.zext %46 : i32 to i64
    %48 = llvm.mul %47, %10 : i64
    %49 = llvm.add %48, %31 : i64
    %50 = llvm.getelementptr %arg1[%49] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
    %51 = llvm.load %50 {alignment = 2 : i64} : !llvm.ptr<1> -> vector<1xf16>
    %52 = llvm.add %38, %24 overflow<nsw> : i32
    %53 = llvm.zext %52 : i32 to i64
    %54 = llvm.mul %53, %10 : i64
    %55 = llvm.add %54, %31 : i64
    %56 = llvm.getelementptr %arg1[%55] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
    %57 = llvm.load %56 {alignment = 2 : i64} : !llvm.ptr<1> -> vector<1xf16>
    %58 = llvm.add %38, %18 overflow<nsw> : i32
    %59 = llvm.zext %58 : i32 to i64
    %60 = llvm.mul %59, %10 : i64
    %61 = llvm.add %60, %31 : i64
    %62 = llvm.getelementptr %arg1[%61] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
    %63 = llvm.load %62 {alignment = 2 : i64} : !llvm.ptr<1> -> vector<1xf16>
    %64 = llvm.add %38, %23 overflow<nsw> : i32
    %65 = llvm.zext %64 : i32 to i64
    %66 = llvm.mul %65, %10 : i64
    %67 = llvm.add %66, %31 : i64
    %68 = llvm.getelementptr %arg1[%67] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
    %69 = llvm.load %68 {alignment = 2 : i64} : !llvm.ptr<1> -> vector<1xf16>
    %70 = llvm.add %38, %17 overflow<nsw> : i32
    %71 = llvm.zext %70 : i32 to i64
    %72 = llvm.mul %71, %10 : i64
    %73 = llvm.add %72, %31 : i64
    %74 = llvm.getelementptr %arg1[%73] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
    %75 = llvm.load %74 {alignment = 2 : i64} : !llvm.ptr<1> -> vector<1xf16>
    %76 = llvm.add %38, %16 overflow<nsw> : i32
    %77 = llvm.zext %76 : i32 to i64
    %78 = llvm.mul %77, %10 : i64
    %79 = llvm.add %78, %31 : i64
    %80 = llvm.getelementptr %arg1[%79] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
    %81 = llvm.load %80 {alignment = 2 : i64} : !llvm.ptr<1> -> vector<1xf16>
    %82 = llvm.add %38, %15 overflow<nsw> : i32
    %83 = llvm.zext %82 : i32 to i64
    %84 = llvm.mul %83, %10 : i64
    %85 = llvm.add %84, %31 : i64
    %86 = llvm.getelementptr %arg1[%85] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
    %87 = llvm.load %86 {alignment = 2 : i64} : !llvm.ptr<1> -> vector<1xf16>
    %88 = llvm.extractelement %45[%9 : i64] : vector<1xf16>
    %89 = llvm.extractelement %51[%9 : i64] : vector<1xf16>
    %90 = llvm.extractelement %57[%9 : i64] : vector<1xf16>
    %91 = llvm.extractelement %63[%9 : i64] : vector<1xf16>
    %92 = llvm.extractelement %69[%9 : i64] : vector<1xf16>
    %93 = llvm.extractelement %75[%9 : i64] : vector<1xf16>
    %94 = llvm.extractelement %81[%9 : i64] : vector<1xf16>
    %95 = llvm.extractelement %87[%9 : i64] : vector<1xf16>
    %96 = llvm.extractelement %41[%9 : i64] : vector<8xf16>
    %97 = llvm.insertelement %96, %1[%9 : i64] : vector<1xf16>
    %98 = llvm.extractelement %41[%8 : i64] : vector<8xf16>
    %99 = llvm.insertelement %98, %1[%9 : i64] : vector<1xf16>
    %100 = llvm.extractelement %41[%7 : i64] : vector<8xf16>
    %101 = llvm.insertelement %100, %1[%9 : i64] : vector<1xf16>
    %102 = llvm.extractelement %41[%6 : i64] : vector<8xf16>
    %103 = llvm.insertelement %102, %1[%9 : i64] : vector<1xf16>
    %104 = llvm.extractelement %41[%5 : i64] : vector<8xf16>
    %105 = llvm.insertelement %104, %1[%9 : i64] : vector<1xf16>
    %106 = llvm.extractelement %41[%4 : i64] : vector<8xf16>
    %107 = llvm.insertelement %106, %1[%9 : i64] : vector<1xf16>
    %108 = llvm.extractelement %41[%3 : i64] : vector<8xf16>
    %109 = llvm.insertelement %108, %1[%9 : i64] : vector<1xf16>
    %110 = llvm.extractelement %41[%2 : i64] : vector<8xf16>
    %111 = llvm.insertelement %110, %1[%9 : i64] : vector<1xf16>
    %112 = llvm.insertelement %88, %1[%9 : i64] : vector<1xf16>
    %113 = llvm.insertelement %89, %1[%9 : i64] : vector<1xf16>
    %114 = llvm.insertelement %90, %1[%9 : i64] : vector<1xf16>
    %115 = llvm.insertelement %91, %1[%9 : i64] : vector<1xf16>
    %116 = llvm.insertelement %92, %1[%9 : i64] : vector<1xf16>
    %117 = llvm.insertelement %93, %1[%9 : i64] : vector<1xf16>
    %118 = llvm.insertelement %94, %1[%9 : i64] : vector<1xf16>
    %119 = llvm.insertelement %95, %1[%9 : i64] : vector<1xf16>
    %120 = llvm.extractvalue %35[0, 0, 0, 0, 0] : !llvm.array<1 x array<1 x array<1 x array<1 x array<1 x vector<1xf16>>>>>> 
    %121 = llvm.extractelement %111[%9 : i64] : vector<1xf16>
    %122 = llvm.extractelement %119[%9 : i64] : vector<1xf16>
    %123 = llvm.extractelement %120[%9 : i64] : vector<1xf16>
    %124 = llvm.fpext %121 : f16 to f32
    %125 = llvm.fpext %122 : f16 to f32
    %126 = llvm.fpext %123 : f16 to f32
    %127 = llvm.call @__nv_fmaf(%124, %125, %126) : (f32, f32, f32) -> f32
    %128 = llvm.fptrunc %127 : f32 to f16
    %129 = llvm.insertelement %128, %1[%9 : i64] : vector<1xf16>
    %130 = llvm.extractelement %109[%9 : i64] : vector<1xf16>
    %131 = llvm.extractelement %118[%9 : i64] : vector<1xf16>
    %132 = llvm.extractelement %129[%9 : i64] : vector<1xf16>
    %133 = llvm.fpext %130 : f16 to f32
    %134 = llvm.fpext %131 : f16 to f32
    %135 = llvm.fpext %132 : f16 to f32
    %136 = llvm.call @__nv_fmaf(%133, %134, %135) : (f32, f32, f32) -> f32
    %137 = llvm.fptrunc %136 : f32 to f16
    %138 = llvm.insertelement %137, %1[%9 : i64] : vector<1xf16>
    %139 = llvm.extractelement %107[%9 : i64] : vector<1xf16>
    %140 = llvm.extractelement %117[%9 : i64] : vector<1xf16>
    %141 = llvm.extractelement %138[%9 : i64] : vector<1xf16>
    %142 = llvm.fpext %139 : f16 to f32
    %143 = llvm.fpext %140 : f16 to f32
    %144 = llvm.fpext %141 : f16 to f32
    %145 = llvm.call @__nv_fmaf(%142, %143, %144) : (f32, f32, f32) -> f32
    %146 = llvm.fptrunc %145 : f32 to f16
    %147 = llvm.insertelement %146, %1[%9 : i64] : vector<1xf16>
    %148 = llvm.extractelement %105[%9 : i64] : vector<1xf16>
    %149 = llvm.extractelement %116[%9 : i64] : vector<1xf16>
    %150 = llvm.extractelement %147[%9 : i64] : vector<1xf16>
    %151 = llvm.fpext %148 : f16 to f32
    %152 = llvm.fpext %149 : f16 to f32
    %153 = llvm.fpext %150 : f16 to f32
    %154 = llvm.call @__nv_fmaf(%151, %152, %153) : (f32, f32, f32) -> f32
    %155 = llvm.fptrunc %154 : f32 to f16
    %156 = llvm.insertelement %155, %1[%9 : i64] : vector<1xf16>
    %157 = llvm.extractelement %103[%9 : i64] : vector<1xf16>
    %158 = llvm.extractelement %115[%9 : i64] : vector<1xf16>
    %159 = llvm.extractelement %156[%9 : i64] : vector<1xf16>
    %160 = llvm.fpext %157 : f16 to f32
    %161 = llvm.fpext %158 : f16 to f32
    %162 = llvm.fpext %159 : f16 to f32
    %163 = llvm.call @__nv_fmaf(%160, %161, %162) : (f32, f32, f32) -> f32
    %164 = llvm.fptrunc %163 : f32 to f16
    %165 = llvm.insertelement %164, %1[%9 : i64] : vector<1xf16>
    %166 = llvm.extractelement %101[%9 : i64] : vector<1xf16>
    %167 = llvm.extractelement %114[%9 : i64] : vector<1xf16>
    %168 = llvm.extractelement %165[%9 : i64] : vector<1xf16>
    %169 = llvm.fpext %166 : f16 to f32
    %170 = llvm.fpext %167 : f16 to f32
    %171 = llvm.fpext %168 : f16 to f32
    %172 = llvm.call @__nv_fmaf(%169, %170, %171) : (f32, f32, f32) -> f32
    %173 = llvm.fptrunc %172 : f32 to f16
    %174 = llvm.insertelement %173, %1[%9 : i64] : vector<1xf16>
    %175 = llvm.extractelement %99[%9 : i64] : vector<1xf16>
    %176 = llvm.extractelement %113[%9 : i64] : vector<1xf16>
    %177 = llvm.extractelement %174[%9 : i64] : vector<1xf16>
    %178 = llvm.fpext %175 : f16 to f32
    %179 = llvm.fpext %176 : f16 to f32
    %180 = llvm.fpext %177 : f16 to f32
    %181 = llvm.call @__nv_fmaf(%178, %179, %180) : (f32, f32, f32) -> f32
    %182 = llvm.fptrunc %181 : f32 to f16
    %183 = llvm.insertelement %182, %1[%9 : i64] : vector<1xf16>
    %184 = llvm.extractelement %97[%9 : i64] : vector<1xf16>
    %185 = llvm.extractelement %112[%9 : i64] : vector<1xf16>
    %186 = llvm.extractelement %183[%9 : i64] : vector<1xf16>
    %187 = llvm.fpext %184 : f16 to f32
    %188 = llvm.fpext %185 : f16 to f32
    %189 = llvm.fpext %186 : f16 to f32
    %190 = llvm.call @__nv_fmaf(%187, %188, %189) : (f32, f32, f32) -> f32
    %191 = llvm.fptrunc %190 : f32 to f16
    %192 = llvm.insertelement %191, %1[%9 : i64] : vector<1xf16>
    %193 = llvm.insertvalue %192, %13[0, 0, 0, 0, 0] : !llvm.array<1 x array<1 x array<1 x array<1 x array<1 x vector<1xf16>>>>>> 
    %194 = llvm.add %34, %25 : i32
    llvm.br ^bb1(%194, %193 : i32, !llvm.array<1 x array<1 x array<1 x array<1 x array<1 x vector<1xf16>>>>>>)
  ^bb3:  // pred: ^bb1
    %195 = llvm.extractvalue %35[0, 0, 0, 0, 0] : !llvm.array<1 x array<1 x array<1 x array<1 x array<1 x vector<1xf16>>>>>> 
    %196 = llvm.extractelement %195[%9 : i64] : vector<1xf16>
    %197 = llvm.fadd %196, %14 : f16
    %198 = llvm.bitcast %197 : f16 to i16
    %199 = llvm.zext %198 : i16 to i32
    %200 = llvm.sub %25, %25 : i32
    %201 = llvm.lshr %0, %200 : i32
    %202 = llvm.sub %25, %26 : i32
    %203 = nvvm.shfl.sync bfly %201, %199, %26, %202 : i32 -> i32
    %204 = llvm.trunc %203 : i32 to i16
    %205 = llvm.bitcast %204 : i16 to f16
    %206 = llvm.fadd %197, %205 : f16
    %207 = llvm.bitcast %206 : f16 to i16
    %208 = llvm.zext %207 : i16 to i32
    %209 = llvm.sub %25, %25 : i32
    %210 = llvm.lshr %0, %209 : i32
    %211 = llvm.sub %25, %26 : i32
    %212 = nvvm.shfl.sync bfly %210, %208, %24, %211 : i32 -> i32
    %213 = llvm.trunc %212 : i32 to i16
    %214 = llvm.bitcast %213 : i16 to f16
    %215 = llvm.fadd %206, %214 : f16
    %216 = llvm.bitcast %215 : f16 to i16
    %217 = llvm.zext %216 : i16 to i32
    %218 = llvm.sub %25, %25 : i32
    %219 = llvm.lshr %0, %218 : i32
    %220 = llvm.sub %25, %26 : i32
    %221 = nvvm.shfl.sync bfly %219, %217, %23, %220 : i32 -> i32
    %222 = llvm.trunc %221 : i32 to i16
    %223 = llvm.bitcast %222 : i16 to f16
    %224 = llvm.fadd %215, %223 : f16
    %225 = llvm.bitcast %224 : f16 to i16
    %226 = llvm.zext %225 : i16 to i32
    %227 = llvm.sub %25, %25 : i32
    %228 = llvm.lshr %0, %227 : i32
    %229 = llvm.sub %25, %26 : i32
    %230 = nvvm.shfl.sync bfly %228, %226, %22, %229 : i32 -> i32
    %231 = llvm.trunc %230 : i32 to i16
    %232 = llvm.bitcast %231 : i16 to f16
    %233 = llvm.fadd %224, %232 : f16
    %234 = llvm.bitcast %233 : f16 to i16
    %235 = llvm.zext %234 : i16 to i32
    %236 = llvm.sub %25, %25 : i32
    %237 = llvm.lshr %0, %236 : i32
    %238 = llvm.sub %25, %26 : i32
    %239 = nvvm.shfl.sync bfly %237, %235, %21, %238 : i32 -> i32
    %240 = llvm.trunc %239 : i32 to i16
    %241 = llvm.bitcast %240 : i16 to f16
    %242 = llvm.fadd %233, %241 : f16
    %243 = llvm.fadd %242, %14 : f16
    %244 = llvm.insertelement %243, %1[%20 : i32] : vector<1xf16>
    %245 = llvm.icmp "eq" %32, %20 : i32
    llvm.cond_br %245, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %246 = llvm.getelementptr %arg2[%31] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
    llvm.store %244, %246 {alignment = 2 : i64} : vector<1xf16>, !llvm.ptr<1>
    llvm.br ^bb5
  ^bb5:  // 2 preds: ^bb3, ^bb4
    llvm.return
  }
}

