(module
    (import "dummy" "memory" (memory 1))
    ;; Transpose f32 2x2 matrix sequentially (contiguous layout) in place
    ;; Load four f32 numbers from location 0
    (func
        (export  "transpose_f32x2x2")

        i32.const 0
        i32.const 0
        f32.load offset=4 align=4
        i32.const 0
        i32.const 0
        f32.load offset=8 align=4

        f32.store offset=4 align=4
        f32.store offset=8 align=4
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
    ;; Transpose f32 4x4 matrix sequentially (contiguous layout) in place
    ;; Load sixteen f32 numbers from location indicated by first paramter and
    ;; store transposed matrix at location indicated by second parameter
    (func
        (export  "transpose_f32x4x4")
        (param i32 i32)

        ;; (1, 2, 3)
        get_local 1
        get_local 0
        f32.load offset=4 align=4
        get_local 1
        get_local 0
        f32.load offset=8 align=4
        get_local 1
        get_local 0
        f32.load offset=12 align=4
        ;; (4, 8, 12)
        get_local 1
        get_local 0
        f32.load offset=16 align=4
        get_local 1
        get_local 0
        f32.load offset=32 align=4
        get_local 1
        get_local 0
        f32.load offset=48 align=4
        ;; (6, 7)
        get_local 1
        get_local 0
        f32.load offset=24 align=4
        get_local 1
        get_local 0
        f32.load offset=28 align=4
        ;; (9, 13)
        get_local 1
        get_local 0
        f32.load offset=36 align=4
        get_local 1
        get_local 0
        f32.load offset=52 align=4
        ;; (11) and (14)
        get_local 1
        get_local 0
        f32.load offset=44 align=4
        get_local 1
        get_local 0
        f32.load offset=56 align=4

        ;; Store in reverse order
        ;; (14) and (11)
        f32.store offset=56 align=4
        f32.store offset=44 align=4
        ;; (7, 6)
        f32.store offset=28 align=4
        f32.store offset=24 align=4
        ;; (13, 9)
        f32.store offset=52 align=4
        f32.store offset=36 align=4
        ;; (3, 2, 1)
        f32.store offset=12 align=4
        f32.store offset=8 align=4
        f32.store offset=4 align=4
        ;; (12, 8, 4)
        f32.store offset=48 align=4
        f32.store offset=32 align=4
        f32.store offset=16 align=4
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
    ;; Multiply two f32 4x4 matrices sequentially (contiguous layout)
    ;; First operand is at location 0, second -- at location 64 and result is
    ;; at location 128
    (func
        (export  "mul_f32x4x4")
        i32.const 64
        i32.const 192
        call 2 ;; TODO explicit indices in function definitions
    )
    ;; f32x4.mul vec1 vec2
)
