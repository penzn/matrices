(module
    (import "dummy" "memory" (memory 1))

    ;; Scalar transpose a square matrix (contiguous layout) of 32-bit elements in place
    ;;
    ;; @param $ptr source and destination -- byte index
    ;; @param $sz size of the matrix (number of elements in a row)
    (func $tr
        (export "transpose_32")
        (param $ptr i32)
        (param $sz i32)
        (local $i i32)
        (local $j i32)
        (local $src_ptr i32)
        (local $dst_ptr i32)

        (set_local $j (i32.const 0))

        (block $outer
            (loop $outer_top
                (br_if $outer (i32.eq (get_local $j) (i32.sub (get_local $sz) (i32.const 1))))
                (set_local $i (i32.add (get_local $j) (i32.const 1)))
                (block $inner
                    (loop $inner_top
                        (br_if $inner (i32.eq (get_local $i) (get_local $sz)))

                        (set_local $src_ptr
                            (i32.add (get_local $ptr)
                                (i32.mul (i32.const 4)
                                    (i32.add (i32.mul (get_local $i) (get_local $sz)) (get_local $j))
                                )
                            )
                        )
                        (set_local $dst_ptr
                            (i32.add (get_local $ptr)
                                (i32.mul (i32.const 4)
                                    (i32.add (i32.mul (get_local $j) (get_local $sz)) (get_local $i))
                                )
                            )
                        )

                        get_local $dst_ptr
                        get_local $src_ptr
                        i32.load offset=0 align=4
                        get_local $src_ptr
                        get_local $dst_ptr
                        i32.load offset=0 align=4
                        i32.store offset=0 align=4
                        i32.store offset=0 align=4

                        (set_local $i (i32.add (get_local $i) (i32.const 1)))
                        (br $inner_top)
                    )
                )
                (set_local $j (i32.add (get_local $j) (i32.const 1)))
                (br $outer_top)
            )
        )
    )

    ;; Dot product of two f32 vectors
    ;;
    ;; @param $src1 memory location of the first vector
    ;; @param $src2 memory location of the second vector (transposed)
    ;; @param $sz vector length (# of elements)
    ;; @return dot product value
    (func $df
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

    ;; Multiply f32 vector and a square f32 matrix
    ;;
    ;; @param $v vector memory location
    ;; @param $m memory location of transposed matrix
    ;; @param $dst memory location for the result
    ;; @param $sz size of both
    (func $vmf
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

                (f32.store (get_local $dptr) (call $df (get_local $v) (get_local $mptr) (get_local $sz)))

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

                (call $vmf
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

    ;; Dot product of two f32 vectors, SIMD version
    ;;
    ;; @param $src1 memory location of the first vector
    ;; @param $src2 memory location of the second vector (transposed)
    ;; @param $sz vector length (# of elements)
    ;; @return dot product value
    (func $df_vec
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

    ;; Multiply f32 vector and a square matrix, SIMD version
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

                (f32.store (get_local $dptr) (call $df_vec (get_local $v) (get_local $mptr) (get_local $sz)))

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

    ;; Dot product of two i32 vectors
    ;;
    ;; @param $src1 memory location of the first vector
    ;; @param $src2 memory location of the second vector (transposed)
    ;; @param $sz vector length (# of elements)
    ;; @return dot product value
    (func $di
       (export "dot_i32")
       (param $src1 i32)
       (param $src2 i32)
       (param $sz i32)
       (result i32)

       (local $off i32)
       (local $len i32)
       (local $sum i32)

       (set_local $sum (i32.const 0))
       (set_local $off (i32.const 0))
       (set_local $len (i32.mul (get_local $sz) (i32.const 4)))

       (block $loop
           (loop $loop_top
               (br_if $loop (i32.eq (get_local $off) (get_local $len)))

               (set_local $sum
                   (i32.add (get_local $sum)
                       (i32.mul
                           (i32.load (i32.add (get_local $src1) (get_local $off)))
                           (i32.load (i32.add (get_local $src2) (get_local $off)))
                       )
                   )
               )
               (set_local $off (i32.add (get_local $off) (i32.const 4)))
               (br $loop_top)
           )
       )

       get_local $sum
    )

    ;; Multiply i32 vector and a square i32 matrix
    ;;
    ;; @param $v vector memory location
    ;; @param $m memory location of transposed matrix
    ;; @param $dst memory location for the result
    ;; @param $sz size of both
    (func $vmi
        (export "vmmul_i32")
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

                (i32.store (get_local $dptr) (call $di (get_local $v) (get_local $mptr) (get_local $sz)))

                (set_local $mptr (i32.add (get_local $mptr) (get_local $rowlen)))
                (set_local $dptr (i32.add (get_local $dptr) (i32.const 4)))
                (set_local $i (i32.add (get_local $i) (i32.const 1)))
                (br $loop_top)
            )
        )
    )

    ;; Scalar multiplication of square i32 matrices (contiguous layout)
    ;;
    ;; @param $src1 location (byte index) of the first operand
    ;; @param $src2 location of the second operand -- transposed
    ;; @param $dst location where to write the result
    ;; @param $sz size of the matrix (number of elements in a row)
    (func
        (export "multiply_i32")
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

                (call $vmi
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

    ;; Dot product of two i32 vectors, SIMD version
    ;;
    ;; @param $src1 memory location of the first vector
    ;; @param $src2 memory location of the second vector (transposed)
    ;; @param $sz vector length (# of elements)
    ;; @return dot product value
    (func $di_vec
       (export "dot_i32_simd")
       (param $src1 i32)
       (param $src2 i32)
       (param $sz i32)
       (result i32)

       (local $i i32)
       (local $ptr1 i32)
       (local $ptr2 i32)
       (local $N i32)
       (local $sum i32)
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
                   (i32x4.add
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
           (i32.add (i32x4.extract_lane 0 (get_local $v))
               (i32.add (i32x4.extract_lane 1 (get_local $v))
                   (i32.add (i32x4.extract_lane 2 (get_local $v))
                       (i32x4.extract_lane 3 (get_local $v))
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
                   (i32.add (get_local $sum)
                       (i32.mul (i32.load (get_local $ptr1)) (i32.load (get_local $ptr2)))
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

    ;; Multiply i32 vector and a square matrix, SIMD version
    ;;
    ;; @param $v vector memory location
    ;; @param $m memory location of transposed matrix
    ;; @param $dst memory location for the result
    ;; @param $sz size of both
    (func $vmi_vec
        (export "vmmul_i32_simd")
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

                (i32.store (get_local $dptr) (call $di_vec (get_local $v) (get_local $mptr) (get_local $sz)))

                (set_local $mptr (i32.add (get_local $mptr) (get_local $rowlen)))
                (set_local $dptr (i32.add (get_local $dptr) (i32.const 4)))
                (set_local $i (i32.add (get_local $i) (i32.const 1)))
                (br $loop_top)
            )
        )
    )

    ;; Vector multiplication of square i32 matrices (contiguous layout)
    ;;
    ;; @param $src1 location (byte index) of the first operand
    ;; @param $src2 location of the second operand -- transposed
    ;; @param $dst location where to write the result
    ;; @param $sz size of the matrix (number of elements in a row)
    (func
        (export "multiply_i32_simd")
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

                (call $vmi_vec
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

    ;; Test harness for scalar transpose for 32-bit elements
    ;;
    ;; @param $ptr source and destination -- byte index
    ;; @param $sz size of the matrix (number of elements in a row)
    ;; @param $cnt how many times to perform the transposition
    (func
        (export "test_transpose_32")
        (param $ptr i32)
        (param $sz i32)
        (param $cnt i32)
        (local $i i32)

        (set_local $i (i32.const 0))

        (block $loop
            (loop $loop_top
                (br_if $loop (i32.eq (get_local $i) (get_local $cnt)))

                (call $tr (get_local $ptr) (get_local $sz))

                (set_local $i (i32.add (get_local $i) (i32.const 1)))
                (br $loop_top)
            )
        )
    )
    ;; Swap 2x2 matrix
    (func $swap
        (param $a1 i32)
        (param $a2 i32)
        (param $b1 i32)
        (param $b2 i32)

    )
    ;; SIMD transpose a square matrix (contiguous layout) of 32-bit elements in place
    ;;
    ;; @param $ptr source and destination -- byte index
    ;; @param $sz size of the matrix (number of elements in a row)
    (func $vtr
        (export "simd_transpose_32")
        (param $ptr i32)
        (param $sz i32)
        (local $i i32)
        (local $j i32)
        (local $src_ptr i32)
        (local $dst_ptr i32)
        (local $sz_norm i32)
        (local $v v128)

        (set_local $sz_norm (i32.mul (i32.const 2) (i32.div_u (get_local $sz) (i32.const 2))))
        (set_local $j (i32.const 0))

        (block $outer
            (loop $outer_top
                ;;(br_if $outer (i32.eq (get_local $j) (i32.sub (get_local $sz_norm) (i32.const 1))))
                ;;(set_local $i (i32.add (get_local $j) (i32.const 1)))
                (br_if $outer (i32.eq (get_local $j) (get_local $sz_norm)))
                (set_local $i (get_local $j))
                (block $inner
                    (loop $inner_top
                        ;; FIXME sometimes $i == $sz_norm at the beginning, would that affect execution?
                        ;; FIXME and if that is a bug in the runtime, how to demonstrate that?
                        (br_if $inner (i32.eq (get_local $i) (get_local $sz_norm)))

                        (set_local $src_ptr
                            (i32.add (get_local $ptr)
                                (i32.mul (i32.const 4)
                                    (i32.add (i32.mul (get_local $i) (get_local $sz)) (get_local $j))
                                )
                            )
                        )

                        ;; Tiles are handled differently based on whether they are on the diagonal or not
                        (if (i32.eq (get_local $i) (get_local $j))
                            (then
                                get_local $src_ptr
                                v128.load offset=0 align=4 ;; first line of the tile
                                get_local $src_ptr
                                get_local $sz
                                i32.const 4
                                i32.mul
                                i32.add
                                v128.load offset=0 align=4 ;; second line of the tile
                                v8x16.shuffle 0x03020100 0x13121110 0x07060504 0x17161514 ;; Transposed tile inline
                                ;;v8x16.shuffle 0x03020100 0x07060504 0x0B0A0908 0x0F0E0D0C
                                set_local $v
                            )
                            (else
                                (set_local $dst_ptr
                                    (i32.add (get_local $ptr)
                                        (i32.mul (i32.const 4)
                                            (i32.add (i32.mul (get_local $j) (get_local $sz)) (get_local $i))
                                        )
                                    )
                                )

                                get_local $dst_ptr
                                get_local $src_ptr
                                i32.load offset=0 align=4
                                get_local $src_ptr
                                get_local $dst_ptr
                                i32.load offset=0 align=4
                                i32.store offset=0 align=4
                                i32.store offset=0 align=4

                            )
                        )

                        (set_local $i (i32.add (get_local $i) (i32.const 1)))
                        (br $inner_top)
                    )
                )

                ;; For odd sizes, we have one more element at the end
                (block $onemore
                    (br_if $onemore (i32.eq (get_local $sz) (get_local $sz_norm)))

                    (set_local $src_ptr
                        (i32.add (get_local $ptr)
                            (i32.mul (i32.const 4)
                                (i32.add (i32.mul (get_local $i) (get_local $sz)) (get_local $j))
                            )
                        )
                    )
                    (set_local $dst_ptr
                        (i32.add (get_local $ptr)
                            (i32.mul (i32.const 4)
                                (i32.add (i32.mul (get_local $j) (get_local $sz)) (get_local $i))
                            )
                        )
                    )

                    get_local $dst_ptr
                    get_local $src_ptr
                    i32.load offset=0 align=4
                    get_local $src_ptr
                    get_local $dst_ptr
                    i32.load offset=0 align=4
                    i32.store offset=0 align=4
                    i32.store offset=0 align=4
                )

                (set_local $j (i32.add (get_local $j) (i32.const 1)))
                (br $outer_top)
            )
        )
    )

    ;; Vector transpose f32 2x2 matrix (contiguous layout) in place
    ;; Load a vector of four f32 numbers from location indicated by $ptr
    (func $vt2x2
        (export  "simd_transpose_f32x2x2")
        (param $ptr i32)

        get_local $ptr
        get_local $ptr
        v128.load offset=0 align=4
        v128.const i32 0 0 0 0
        v8x16.shuffle 0x03020100 0x0B0A0908 0x07060504 0x0F0E0D0C
        ;;v8x16.shuffle 0x03020100 0x07060504 0x0B0A0908 0x0F0E0D0C
        v128.store offset=0 align=4
    )
    (func
        (export "test_vec_transpose_2x2")
        (param $ptr i32)
        (param $cnt i32)
        (local $i i32)

        (set_local $i (i32.const 0))

        (block $loop
            (loop $loop_top
                (br_if $loop (i32.eq (get_local $i) (get_local $cnt)))

                (call $vt2x2 (get_local $ptr))

                (set_local $i (i32.add (get_local $i) (i32.const 1)))
                (br $loop_top)
            )
        )
    )
    ;; Vector transpose f32 4x4 matrix (contiguous layout) in place
    ;; Load four f32x4 vectors from location indicated by first paramter and
    ;; store transposed matrix at location indicated by second parameter
    (func $vt4x4
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
    (func
        (export "test_vec_transpose_4x4")
        (param $ptr i32)
        (param $cnt i32)
        (local $i i32)

        (set_local $i (i32.const 0))

        (block $loop
            (loop $loop_top
                (br_if $loop (i32.eq (get_local $i) (get_local $cnt)))

                (call $vt4x4 (get_local $ptr) (get_local $ptr))

                (set_local $i (i32.add (get_local $i) (i32.const 1)))
                (br $loop_top)
            )
        )
    )
)
