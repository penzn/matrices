(module
    (import "dummy" "memory" (memory 1))
    ;; Scalar transpose a square f32 matrix (contiguous layout) in place
    ;;
    ;; @param $ptr source and destination -- byte index
    ;; @param size (width) of the matrix in elements
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
            (block $outer_top
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
