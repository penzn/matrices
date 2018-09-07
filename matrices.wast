(module
    (import "dummy" "memory" (memory 1))
    ;; Scalar transpose a square f32 matrix (contiguous layout) in place
    ;;
    ;; @param $ptr source and destination -- byte index
    ;; @param $sz size of the matrix (number of elements in a row)
    (func
        (export  "transpose_f32")
        (param $ptr i32)
        (param $sz i32)
        (local $i i32)
        (local $j i32)
        (local $src_ptr i32)
        (local $dst_ptr i32)

        (set_local $i (i32.const 0))
        (set_local $j (i32.const 0))
        (set_local $src_ptr (get_local $ptr))

        (block $outer
            (loop $outer_top
                (br_if $outer (i32.eq (get_local $j) (get_local $sz)))
                (set_local $dst_ptr (i32.add (get_local $ptr) (i32.mul (get_local $j) (i32.const 4)))) ;; Start ptr is row index, TODO optimize mul
                (block $inner
                    (loop $inner_top
                        (br_if $inner (i32.eq (get_local $i) (get_local $sz)))

                        ;; TODO don't swap when i == j

                        get_local $dst_ptr
                        get_local $src_ptr
                        f32.load offset=0 align=4
                        get_local $src_ptr
                        get_local $dst_ptr
                        f32.load offset=0 align=4
                        f32.store offset=0 align=4
                        f32.store offset=0 align=4

                        (set_local $dst_ptr (i32.add (get_local $dst_ptr) (i32.mul (i32.const 4) (get_local $sz)))) ;; TODO optimize mul
                        (set_local $src_ptr (i32.add (get_local $src_ptr) (i32.const 4)))
                        (set_local $i (i32.add (get_local $i) (i32.const 1)))
                        (br $inner_top)
                    )
                )
                (set_local $j (i32.add (get_local $j) (i32.const 1)))
                (br $outer_top)
            )
        )
    )
    ;; Dot product of two vectors
    ;;
    ;; @param $src1 memory location of the first vector
    ;; @param $src2 memory location of the second vector (transposed)
    ;; @param $sz vector length (# of elements)
    ;; @return dot product value
    (func $d
       (export "dot_f32")
       (param $src1 i32)
       (param $src2 i32)
       (param $sz i32)
       (result f32)

       (local $off i32)
       (local $len i32)
       (local $sum f32)

       (set_local $sum (f32.const 0))
       (set_local $off (i32.const 0))
       (set_local $len (i32.mul (get_local $sz) (i32.const 4)))

       (block $loop
           (loop $loop_top
               (br_if $loop (i32.eq (get_local $off) (get_local $len)))

               (set_local $sum
                   (f32.add (get_local $sum)
                       (f32.mul
                           (f32.load (i32.add (get_local $src1) (get_local $off)))
                           (f32.load (i32.add (get_local $src2) (get_local $off)))
                       )
                   )
               )
               (set_local $off (i32.add (get_local $off) (i32.const 4)))
               (br $loop_top)
           )
       )

       get_local $sum
    )

    ;; Multiply vector and a square matrix
    ;;
    ;; @param $v vector memory location
    ;; @param $m memory location of transposed matrix
    ;; @param $dst memory location for the result
    ;; @param $sz size of both
    (func $vm
        (export "vmmul_f32")
        (param $v i32)
        (param $m i32)
        (param $dst i32)
        (param $sz i32)

        (local $i i32)
        (local $mptr i32)
        (local $dptr i32)
        (local $rowlen i32)

        (set_local $i (i32.const 0))
        (set_local $mptr (get_local $m))
        (set_local $dptr (get_local $dst))
        (set_local $rowlen (i32.mul (get_local $sz) (i32.const 4)))

        (block $loop
            (loop $loop_top
                (br_if $loop (i32.eq (get_local $i) (get_local $sz)))

                (f32.store (get_local $dptr) (call $d (get_local $v) (get_local $mptr) (get_local $sz)))

                (set_local $mptr (i32.add (get_local $mptr) (get_local $rowlen)))
                (set_local $dptr (i32.add (get_local $dptr) (i32.const 4)))
                (set_local $i (i32.add (get_local $i) (i32.const 1)))
                (br $loop_top)
            )
        )
    )

    ;; Scalar multiplication of square f32 matrices (contiguous layout)
    ;;
    ;; @param $src1 location (byte index) of the first operand
    ;; @param $src2 location of the second operand -- transposed
    ;; @param $dst location where to write the result
    ;; @param $sz size of the matrix (number of elements in a row)
    (func
        (export "multiply_f32")
        (param $src1 i32)
        (param $src2 i32)
        (param $dst i32)
        (param $sz i32)

        (local $i i32)
        (local $off i32)
        (local $rowlen i32)

        (set_local $i (i32.const 0))
        (set_local $off (i32.const 0))
        (set_local $rowlen (i32.mul (get_local $sz) (i32.const 4)))

        (block $loop
            (loop $loop_top
                (br_if $loop (i32.eq (get_local $i) (get_local $sz)))

                (call $vm
                    (i32.add (get_local $src1) (get_local $off))
                    (get_local $src2)
                    (i32.add (get_local $dst) (get_local $off)) (get_local $sz)
                )

                (set_local $off (i32.add (get_local $off) (get_local $rowlen)))
                (set_local $i (i32.add (get_local $i) (i32.const 1)))
                (br $loop_top)
            )
        )
    )

    ;; Dot product of two vectors, SIMD version
    ;;
    ;; @param $src1 memory location of the first vector
    ;; @param $src2 memory location of the second vector (transposed)
    ;; @param $sz vector length (# of elements)
    ;; @return dot product value
    (func $d_vec
       (export "dot_f32_simd")
       (param $src1 i32)
       (param $src2 i32)
       (param $sz i32)
       (result f32)

       (local $i i32)
       (local $ptr1 i32)
       (local $ptr2 i32)
       (local $N i32)
       (local $sum f32)
       (local $v v128)

       (set_local $v (v128.const i32 0 0 0 0))
       (set_local $ptr1 (get_local $src1))
       (set_local $ptr2 (get_local $src2))

       (set_local $i (i32.const 0))
       (set_local $N (i32.div_u (get_local $sz) (i32.const 4)))

       (block $loop1
           (loop $loop1_top
               (br_if $loop1 (i32.eq (get_local $i) (get_local $N)))

               (set_local $v
                   (f32x4.add
                       (get_local $v)
                       (f32x4.mul
                           (v128.load align=4 (get_local $ptr1))
                           (v128.load align=4 (get_local $ptr2))
                       )
                   )
               )

               (set_local $ptr1 (i32.add (get_local $ptr1) (i32.const 16)))
               (set_local $ptr2 (i32.add (get_local $ptr2) (i32.const 16)))
               (set_local $i (i32.add (get_local $i) (i32.const 1)))
               (br $loop1_top)
           )
       )

       (set_local $sum
           (f32.add (f32x4.extract_lane 0 (get_local $v))
               (f32.add (f32x4.extract_lane 1 (get_local $v))
                   (f32.add (f32x4.extract_lane 2 (get_local $v))
                       (f32x4.extract_lane 3 (get_local $v))
                   )
               )
           )
       )

       (set_local $i (i32.const 0))
       (set_local $N (i32.rem_u (get_local $sz) (i32.const 4)))

       (block $loop2
           (loop $loop2_top
               (br_if $loop2 (i32.eq (get_local $i) (get_local $N)))

               (set_local $sum
                   (f32.add (get_local $sum)
                       (f32.mul (f32.load (get_local $ptr1)) (f32.load (get_local $ptr2)))
                   )
               )

               (set_local $ptr1 (i32.add (get_local $ptr1) (i32.const 4)))
               (set_local $ptr2 (i32.add (get_local $ptr2) (i32.const 4)))
               (set_local $i (i32.add (get_local $i) (i32.const 1)))
               (br $loop2_top)
           )
       )

       get_local $sum
    )

    ;; Multiply vector and a square matrix, SIMD version
    ;;
    ;; @param $v vector memory location
    ;; @param $m memory location of transposed matrix
    ;; @param $dst memory location for the result
    ;; @param $sz size of both
    (func $vm_vec
        (export "vmmul_f32_simd")
        (param $v i32)
        (param $m i32)
        (param $dst i32)
        (param $sz i32)

        (local $i i32)
        (local $mptr i32)
        (local $dptr i32)
        (local $rowlen i32)

        (set_local $i (i32.const 0))
        (set_local $mptr (get_local $m))
        (set_local $dptr (get_local $dst))
        (set_local $rowlen (i32.mul (get_local $sz) (i32.const 4)))

        (block $loop
            (loop $loop_top
                (br_if $loop (i32.eq (get_local $i) (get_local $sz)))

                (f32.store (get_local $dptr) (call $d_vec (get_local $v) (get_local $mptr) (get_local $sz)))

                (set_local $mptr (i32.add (get_local $mptr) (get_local $rowlen)))
                (set_local $dptr (i32.add (get_local $dptr) (i32.const 4)))
                (set_local $i (i32.add (get_local $i) (i32.const 1)))
                (br $loop_top)
            )
        )
    )

    ;; Vector multiplication of square f32 matrices (contiguous layout)
    ;;
    ;; @param $src1 location (byte index) of the first operand
    ;; @param $src2 location of the second operand -- transposed
    ;; @param $dst location where to write the result
    ;; @param $sz size of the matrix (number of elements in a row)
    (func
        (export "multiply_f32_simd")
        (param $src1 i32)
        (param $src2 i32)
        (param $dst i32)
        (param $sz i32)

        (local $i i32)
        (local $off i32)
        (local $rowlen i32)

        (set_local $i (i32.const 0))
        (set_local $off (i32.const 0))
        (set_local $rowlen (i32.mul (get_local $sz) (i32.const 4)))

        (block $loop
            (loop $loop_top
                (br_if $loop (i32.eq (get_local $i) (get_local $sz)))

                (call $vm_vec
                    (i32.add (get_local $src1) (get_local $off))
                    (get_local $src2)
                    (i32.add (get_local $dst) (get_local $off)) (get_local $sz)
                )

                (set_local $off (i32.add (get_local $off) (get_local $rowlen)))
                (set_local $i (i32.add (get_local $i) (i32.const 1)))
                (br $loop_top)
            )
        )
    )

    ;; Vector transpose f32 2x2 matrix (contiguous layout) in place
    ;; Load a vector of four f32 numbers from location 0
    (func
        (export  "simd_transpose_f32x2x2")

        i32.const 0
        i32.const 0
        v128.load offset=0 align=4
        v128.const i32 0 0 0 0
        v8x16.shuffle 0x03020100 0x0B0A0908 0x07060504 0x0F0E0D0C
        ;;v8x16.shuffle 0x03020100 0x07060504 0x0B0A0908 0x0F0E0D0C
        v128.store offset=0 align=4
    )
    ;; Vector transpose f32 4x4 matrix (contiguous layout) in place
    ;; Load four f32x4 vectors from location indicated by first paramter and
    ;; store transposed matrix at location indicated by second parameter
    (func
        (export  "simd_transpose_f32x4x4")
        (param i32 i32)

	;; Treat 4x4 as four 2x2 tiles -- transpose each one and then swap the
        ;; two that don't occupy the diagonal

        get_local 1
        get_local 0
        v128.load offset=0 align=4
        get_local 0
        v128.load offset=16 align=4
        get_local 1
        get_local 0
        v128.load offset=0 align=4
        get_local 0
        v128.load offset=16 align=4
        v8x16.shuffle 0x03020100 0x13121110 0x0B0A0908 0x1B1A1918
        ;;v8x16.shuffle 0x03020100 0x07060504 0x0B0A0908 0x0F0E0D0C
        v128.store offset=0 align=4
        v8x16.shuffle 0x07060504 0x17161514 0x0F0E0D0C 0x1F1E1D1C
        ;;v8x16.shuffle 0x03020100 0x07060504 0x0B0A0908 0x0F0E0D0C
        v128.store offset=16 align=4

        get_local 1
        get_local 0
        v128.load offset=32 align=4
        get_local 0
        v128.load offset=48 align=4
        get_local 1
        get_local 0
        v128.load offset=32 align=4
        get_local 0
        v128.load offset=48 align=4
        v8x16.shuffle 0x03020100 0x13121110 0x0B0A0908 0x1B1A1918
        ;;v8x16.shuffle 0x03020100 0x07060504 0x0B0A0908 0x0F0E0D0C
        v128.store offset=32 align=4
        v8x16.shuffle 0x07060504 0x17161514 0x0F0E0D0C 0x1F1E1D1C
        ;;v8x16.shuffle 0x03020100 0x07060504 0x0B0A0908 0x0F0E0D0C
        v128.store offset=48 align=4

        ;; Swap the tiles
        get_local 1
        get_local 0
        i64.load offset=8 align=4
        get_local 1
        get_local 0
        i64.load offset=32 align=4
        i64.store offset=8 align=4
        i64.store offset=32 align=4
        get_local 1
        get_local 0
        i64.load offset=24 align=4
        get_local 1
        get_local 0
        i64.load offset=48 align=4
        i64.store offset=24 align=4
        i64.store offset=48 align=4
    )
)
