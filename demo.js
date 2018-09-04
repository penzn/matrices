function print_4(arr) {
  print("2x2 data: ");
  print(arr[0] + " " + arr[1]);
  print(arr[2] + " " + arr[3]);
}

function print_16(arr) {
  print("4x4 data: ");
  print(arr[0] + " " + arr[1] + " " + arr[2] + " " + arr[3]);
  print(arr[4] + " " + arr[5] + " " + arr[6] + " " + arr[7]);
  print(arr[8] + " " + arr[9] + " " + arr[10] + " " + arr[11]);
  print(arr[12] + " " + arr[13] + " " + arr[14] + " " + arr[15]);
}

const N = 1000000;
const INITIAL_SIZE = 1;
const memObj = new WebAssembly.Memory({initial:INITIAL_SIZE});
const module = new WebAssembly.Module(readbuffer('matrices.wasm'));
const instance = new WebAssembly.Instance(module, { "dummy" : { "memory" : memObj } }).exports;
let data = new Float32Array (memObj.buffer);

for (let i = 0; i < 16; ++i) {
  data[i] = i + 1.5;
}

print_4(data);

print("==== 2x2 f32 WASM scalar transpose ====");

var tStart = Date.now();
for (let i = 0; i < N; ++i) {
  instance["transpose_f32"](0, 2);
  instance["transpose_f32"](0, 2);
}
var tEnd = Date.now();
print((2 * N) + " transpositions took " + (tEnd - tStart) + " milliseconds.");

print_4(data);

/*
print("==== 2x2 f32 WASM vector transpose ====");

var tStart = Date.now();
for (let i = 0; i < N; ++i) {
  instance["simd_transpose_f32x2x2"]();
  instance["simd_transpose_f32x2x2"]();
}
var tEnd = Date.now();
print((2 * N) + " transpositions took " + (tEnd - tStart) + " milliseconds.");

print_4(data);

print("\n");

for (let i = 0; i < 16; ++i) {
  data[i] = i + 1.5;
}

*/
print_16(data);

print("==== 4x4 f32 WASM scalar transpose ====");

var tStart = Date.now();
for (let i = 0; i < N; ++i) {
  instance["transpose_f32"](0, 4);
  instance["transpose_f32"](0, 4);
}
var tEnd = Date.now();
print((2 * N) + " transpositions took " + (tEnd - tStart) + " milliseconds.");

print_16(data);

/*
print("==== 4x4 f32 WASM vector transpose ====");

var tStart = Date.now();
for (let i = 0; i < N; ++i) {
  instance["simd_transpose_f32x4x4"](0, 0);
  instance["simd_transpose_f32x4x4"](0, 0);
}
var tEnd = Date.now();
print((2 * N) + " transpositions took " + (tEnd - tStart) + " milliseconds.");

print_16(data);
*/
