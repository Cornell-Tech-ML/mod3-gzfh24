# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py

# 3.1 and 3.2 Diagnostic Output

```
MAP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map, 
/Users/georgehuang2020/Repositories/Cornell/cs 
5781/mod3-gzfh24/minitorch/fast_ops.py (164)  
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /Users/georgehuang2020/Repositories/Cornell/cs 5781/mod3-gzfh24/minitorch/fast_ops.py (164) 
-----------------------------------------------------------------------------|loop #ID
    def _map(                                                                | 
        out: Storage,                                                        | 
        out_shape: Shape,                                                    | 
        out_strides: Strides,                                                | 
        in_storage: Storage,                                                 | 
        in_shape: Shape,                                                     | 
        in_strides: Strides,                                                 | 
    ) -> None:                                                               | 
        # TODO: Implement for Task 3.1.                                      | 
        if np.array_equal(in_strides, out_strides) and np.array_equal(       | 
            in_shape, out_shape                                              | 
        ):                                                                   | 
            for i in prange(len(out)):---------------------------------------| #0
                out[i] = fn(in_storage[i])  # type: ignore                   | 
        else:                                                                | 
            for i in prange(len(out)):---------------------------------------| #1
                out_index: Index = np.empty(MAX_DIMS, dtype=np.int32)        | 
                in_index: Index = np.empty(MAX_DIMS, dtype=np.int32)         | 
                to_index(i, out_shape, out_index)                            | 
                broadcast_index(out_index, out_shape, in_shape, in_index)    | 
                o = index_to_position(out_index, out_strides)                | 
                j = index_to_position(in_index, in_strides)                  | 
                out[o] = fn(in_storage[j])                                   | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #0, #1).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/georgehuang2020/Repositories/Cornell/cs 
5781/mod3-gzfh24/minitorch/fast_ops.py (180) is hoisted out of the parallel loop
 labelled #1 (it will be performed before the loop is executed and reused inside
 the loop):
   Allocation:: out_index: Index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/georgehuang2020/Repositories/Cornell/cs 
5781/mod3-gzfh24/minitorch/fast_ops.py (181) is hoisted out of the parallel loop
 labelled #1 (it will be performed before the loop is executed and reused inside
 the loop):
   Allocation:: in_index: Index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip, 
/Users/georgehuang2020/Repositories/Cornell/cs 
5781/mod3-gzfh24/minitorch/fast_ops.py (214)  
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/georgehuang2020/Repositories/Cornell/cs 5781/mod3-gzfh24/minitorch/fast_ops.py (214) 
---------------------------------------------------------------------------|loop #ID
    def _zip(                                                              | 
        out: Storage,                                                      | 
        out_shape: Shape,                                                  | 
        out_strides: Strides,                                              | 
        a_storage: Storage,                                                | 
        a_shape: Shape,                                                    | 
        a_strides: Strides,                                                | 
        b_storage: Storage,                                                | 
        b_shape: Shape,                                                    | 
        b_strides: Strides,                                                | 
    ) -> None:                                                             | 
        # TODO: Implement for Task 3.1.                                    | 
        if (                                                               | 
            np.array_equal(a_strides, b_strides)                           | 
            and np.array_equal(a_strides, out_strides)                     | 
            and np.array_equal(a_shape, b_shape)                           | 
            and np.array_equal(a_shape, out_shape)                         | 
        ):                                                                 | 
            for i in prange(len(out)):-------------------------------------| #2
                out[i] = fn(a_storage[i], b_storage[i])  # type: ignore    | 
        else:                                                              | 
            for i in prange(len(out)):-------------------------------------| #3
                out_index: Index = np.empty(MAX_DIMS, dtype=np.int32)      | 
                a_index: Index = np.empty(MAX_DIMS, dtype=np.int32)        | 
                b_index: Index = np.empty(MAX_DIMS, dtype=np.int32)        | 
                to_index(i, out_shape, out_index)                          | 
                broadcast_index(out_index, out_shape, a_shape, a_index)    | 
                broadcast_index(out_index, out_shape, b_shape, b_index)    | 
                o = index_to_position(out_index, out_strides)              | 
                a_pos = index_to_position(a_index, a_strides)              | 
                b_pos = index_to_position(b_index, b_strides)              | 
                out[o] = fn(a_storage[a_pos], b_storage[b_pos])            | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #2, #3).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/georgehuang2020/Repositories/Cornell/cs 
5781/mod3-gzfh24/minitorch/fast_ops.py (236) is hoisted out of the parallel loop
 labelled #3 (it will be performed before the loop is executed and reused inside
 the loop):
   Allocation:: out_index: Index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/georgehuang2020/Repositories/Cornell/cs 
5781/mod3-gzfh24/minitorch/fast_ops.py (237) is hoisted out of the parallel loop
 labelled #3 (it will be performed before the loop is executed and reused inside
 the loop):
   Allocation:: a_index: Index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/georgehuang2020/Repositories/Cornell/cs 
5781/mod3-gzfh24/minitorch/fast_ops.py (238) is hoisted out of the parallel loop
 labelled #3 (it will be performed before the loop is executed and reused inside
 the loop):
   Allocation:: b_index: Index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce, 
/Users/georgehuang2020/Repositories/Cornell/cs 
5781/mod3-gzfh24/minitorch/fast_ops.py (271)  
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/georgehuang2020/Repositories/Cornell/cs 5781/mod3-gzfh24/minitorch/fast_ops.py (271) 
---------------------------------------------------------------------|loop #ID
    def _reduce(                                                     | 
        out: Storage,                                                | 
        out_shape: Shape,                                            | 
        out_strides: Strides,                                        | 
        a_storage: Storage,                                          | 
        a_shape: Shape,                                              | 
        a_strides: Strides,                                          | 
        reduce_dim: int,                                             | 
    ) -> None:                                                       | 
        # TODO: Implement for Task 3.1.                              | 
        for i in prange(len(out)):-----------------------------------| #4
            out_index: Index = np.empty(MAX_DIMS, dtype=np.int32)    | 
            reduce_size = a_shape[reduce_dim]                        | 
            to_index(i, out_shape, out_index)                        | 
            o = index_to_position(out_index, out_strides)            | 
            j = index_to_position(out_index, a_strides)              | 
            delta = a_strides[reduce_dim]                            | 
            acc = out[o]                                             | 
            for _ in range(reduce_size):                             | 
                acc = fn(acc, a_storage[j])                          | 
                j += delta                                           | 
            out[o] = acc                                             | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #4).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/georgehuang2020/Repositories/Cornell/cs 
5781/mod3-gzfh24/minitorch/fast_ops.py (282) is hoisted out of the parallel loop
 labelled #4 (it will be performed before the loop is executed and reused inside
 the loop):
   Allocation:: out_index: Index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY
 
================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply, 
/Users/georgehuang2020/Repositories/Cornell/cs 
5781/mod3-gzfh24/minitorch/fast_ops.py (297)  
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /Users/georgehuang2020/Repositories/Cornell/cs 5781/mod3-gzfh24/minitorch/fast_ops.py (297) 
--------------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                                | 
    out: Storage,                                                                           | 
    out_shape: Shape,                                                                       | 
    out_strides: Strides,                                                                   | 
    a_storage: Storage,                                                                     | 
    a_shape: Shape,                                                                         | 
    a_strides: Strides,                                                                     | 
    b_storage: Storage,                                                                     | 
    b_shape: Shape,                                                                         | 
    b_strides: Strides,                                                                     | 
) -> None:                                                                                  | 
    """NUMBA tensor matrix multiply function.                                               | 
                                                                                            | 
    Should work for any tensor shapes that broadcast as long as                             | 
                                                                                            | 
    ```                                                                                     | 
    assert a_shape[-1] == b_shape[-2]                                                       | 
    ```                                                                                     | 
                                                                                            | 
    Optimizations:                                                                          | 
                                                                                            | 
    * Outer loop in parallel                                                                | 
    * No index buffers or function calls                                                    | 
    * Inner loop should have no global writes, 1 multiply.                                  | 
                                                                                            | 
                                                                                            | 
    Args:                                                                                   | 
    ----                                                                                    | 
        out (Storage): storage for `out` tensor                                             | 
        out_shape (Shape): shape for `out` tensor                                           | 
        out_strides (Strides): strides for `out` tensor                                     | 
        a_storage (Storage): storage for `a` tensor                                         | 
        a_shape (Shape): shape for `a` tensor                                               | 
        a_strides (Strides): strides for `a` tensor                                         | 
        b_storage (Storage): storage for `b` tensor                                         | 
        b_shape (Shape): shape for `b` tensor                                               | 
        b_strides (Strides): strides for `b` tensor                                         | 
                                                                                            | 
    Returns:                                                                                | 
    -------                                                                                 | 
        None : Fills in `out`                                                               | 
                                                                                            | 
    """                                                                                     | 
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                                  | 
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                                  | 
                                                                                            | 
    for i in prange(out_shape[0]):----------------------------------------------------------| #5
        for j in range(out_shape[-2]):                                                      | 
            for k in range(out_shape[-1]):                                                  | 
                a_pos = i * a_batch_stride + j * a_strides[-2]                              | 
                b_pos = i * b_batch_stride + k * b_strides[-1]                              | 
                acc = 0.0                                                                   | 
                for _ in range(a_shape[-1]):                                                | 
                    acc += a_storage[a_pos] * b_storage[b_pos]                              | 
                    a_pos += a_strides[-1]                                                  | 
                    b_pos += b_strides[-2]                                                  | 
                out_pos = i * out_strides[0] + j * out_strides[-2] + k * out_strides[-1]    | 
                out[out_pos] = acc                                                          | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #5).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
```

# 3.5

## CPU

50 Points, 100 Hidden Layers, 0.05 Learning Rate

Simple:
```
Epoch 0, Loss 4.797834552557831, Correct 31, Time 14.72060489654541 seconds
Epoch 10, Loss 0.7331655631251182, Correct 49, Time 0.10839509963989258 seconds
Epoch 20, Loss 0.4027477350545535, Correct 50, Time 0.10894036293029785 seconds
Epoch 30, Loss 0.4172670530974163, Correct 50, Time 0.11162042617797852 seconds
Epoch 40, Loss 0.7922193446430601, Correct 50, Time 0.11085867881774902 seconds
Epoch 50, Loss 0.3543053604380908, Correct 50, Time 0.10981464385986328 seconds
Epoch 60, Loss 0.17653241115741028, Correct 50, Time 0.1104576587677002 seconds
Epoch 70, Loss 0.6806783062571256, Correct 50, Time 0.11122584342956543 seconds
Epoch 80, Loss 0.3878730454405987, Correct 50, Time 0.11121225357055664 seconds
Epoch 90, Loss 0.5094155311565558, Correct 50, Time 0.11974930763244629 seconds
Epoch 100, Loss 0.5719250528405918, Correct 50, Time 0.2415931224822998 seconds
Epoch 110, Loss 0.8503446001038358, Correct 50, Time 0.11009883880615234 seconds
Epoch 120, Loss 0.12025076264738159, Correct 50, Time 0.1100623607635498 seconds
Epoch 130, Loss 0.5572287020498835, Correct 50, Time 0.11265087127685547 seconds
Epoch 140, Loss 1.015569888649194, Correct 50, Time 0.10779881477355957 seconds
Epoch 150, Loss 0.3015955339047552, Correct 50, Time 0.11968135833740234 seconds
Epoch 160, Loss 0.21875123913031552, Correct 50, Time 0.1129140853881836 seconds
Epoch 170, Loss 0.004780151804991629, Correct 50, Time 0.10973381996154785 seconds
Epoch 180, Loss 0.023837674719842657, Correct 50, Time 0.10965299606323242 seconds
Epoch 190, Loss 0.22113060898939862, Correct 50, Time 0.10788941383361816 seconds
Epoch 200, Loss 0.026997029599007374, Correct 50, Time 0.18093299865722656 seconds
Epoch 210, Loss 0.8154215198772905, Correct 50, Time 0.10976767539978027 seconds
Epoch 220, Loss 0.0025217717666818538, Correct 50, Time 0.1112370491027832 seconds
Epoch 230, Loss 0.3834410313449468, Correct 50, Time 0.10965466499328613 seconds
Epoch 240, Loss 0.14315906464790698, Correct 50, Time 0.11931276321411133 seconds
Epoch 250, Loss 0.2783725617947184, Correct 50, Time 0.1122891902923584 seconds
Epoch 260, Loss 0.024057123323196768, Correct 50, Time 0.1154475212097168 seconds
Epoch 270, Loss 0.7235941654442023, Correct 50, Time 0.11268401145935059 seconds
Epoch 280, Loss 0.03789729314711075, Correct 50, Time 0.11087203025817871 seconds
Epoch 290, Loss 0.13745352419212495, Correct 50, Time 0.10862851142883301 seconds
Epoch 300, Loss 0.021800989357888427, Correct 50, Time 0.2216024398803711 seconds
Epoch 310, Loss 0.08084119082504114, Correct 50, Time 0.1112518310546875 seconds
Epoch 320, Loss 0.12493229794292303, Correct 50, Time 0.12447762489318848 seconds
Epoch 330, Loss 0.7350065262097699, Correct 50, Time 0.11186695098876953 seconds
Epoch 340, Loss 0.5319449056562693, Correct 50, Time 0.10866475105285645 seconds
Epoch 350, Loss 0.16043296868631804, Correct 50, Time 0.11030936241149902 seconds
Epoch 360, Loss 0.06502830795293259, Correct 50, Time 0.10863876342773438 seconds
Epoch 370, Loss 0.7816902366381814, Correct 50, Time 0.11105513572692871 seconds
Epoch 380, Loss 0.695085283286925, Correct 50, Time 0.10937643051147461 seconds
Epoch 390, Loss 0.14313730307420028, Correct 50, Time 0.11672544479370117 seconds
Epoch 400, Loss 0.0068636892620963855, Correct 50, Time 0.21210813522338867 seconds
Epoch 410, Loss 0.005223928735646833, Correct 50, Time 0.11103701591491699 seconds
Epoch 420, Loss 0.0027399610204792515, Correct 50, Time 0.11361360549926758 seconds
Epoch 430, Loss 0.0006485466149100242, Correct 50, Time 0.10899090766906738 seconds
Epoch 440, Loss 0.037392285248913616, Correct 50, Time 0.11064839363098145 seconds
Epoch 450, Loss 0.4656421328990091, Correct 50, Time 0.11592555046081543 seconds
Epoch 460, Loss 0.058512034879396206, Correct 50, Time 0.11067891120910645 seconds
Epoch 470, Loss 0.07140975180156543, Correct 50, Time 0.10875272750854492 seconds
Epoch 480, Loss 0.7889583450735346, Correct 50, Time 0.10908675193786621 seconds
Epoch 490, Loss 0.07884010827454896, Correct 50, Time 0.10940313339233398 seconds
Average Time Per Epoch: 0.12570105430358397 s
```

Split:
```
Epoch 0, Loss 7.7003080116553955, Correct 19, Time 14.402568578720093 seconds
Epoch 10, Loss 6.419580441982268, Correct 39, Time 0.10736823081970215 seconds
Epoch 20, Loss 5.562671244885048, Correct 40, Time 0.19881033897399902 seconds
Epoch 30, Loss 5.543321606753514, Correct 43, Time 0.13498187065124512 seconds
Epoch 40, Loss 3.970722033357078, Correct 44, Time 0.11057615280151367 seconds
Epoch 50, Loss 2.1497246482734127, Correct 44, Time 0.10727524757385254 seconds
Epoch 60, Loss 2.491834064823838, Correct 49, Time 0.10898685455322266 seconds
Epoch 70, Loss 3.35654902675607, Correct 47, Time 0.1086270809173584 seconds
Epoch 80, Loss 1.4828607588860843, Correct 47, Time 0.10882306098937988 seconds
Epoch 90, Loss 3.6783634700475596, Correct 46, Time 0.10706520080566406 seconds
Epoch 100, Loss 1.8255474696005665, Correct 49, Time 0.10769891738891602 seconds
Epoch 110, Loss 0.9046225066303399, Correct 48, Time 0.11170172691345215 seconds
Epoch 120, Loss 1.5718232759086355, Correct 46, Time 0.20421862602233887 seconds
Epoch 130, Loss 1.0221295939476782, Correct 49, Time 0.24544811248779297 seconds
Epoch 140, Loss 0.6271635165270344, Correct 49, Time 0.10858869552612305 seconds
Epoch 150, Loss 0.6734553115218632, Correct 49, Time 0.11069297790527344 seconds
Epoch 160, Loss 1.7724174814066482, Correct 50, Time 0.10895466804504395 seconds
Epoch 170, Loss 0.678933983276971, Correct 48, Time 0.10767984390258789 seconds
Epoch 180, Loss 1.0688059281347624, Correct 49, Time 0.10972452163696289 seconds
Epoch 190, Loss 1.7187148443502325, Correct 49, Time 0.1111001968383789 seconds
Epoch 200, Loss 0.797677434437857, Correct 48, Time 0.1092538833618164 seconds
Epoch 210, Loss 1.4266170577318138, Correct 50, Time 0.10723519325256348 seconds
Epoch 220, Loss 0.43660805374618167, Correct 48, Time 0.18929171562194824 seconds
Epoch 230, Loss 2.6933913564353684, Correct 49, Time 0.2553117275238037 seconds
Epoch 240, Loss 2.364408031935858, Correct 48, Time 0.1120002269744873 seconds
Epoch 250, Loss 3.196775813398183, Correct 49, Time 0.10995912551879883 seconds
Epoch 260, Loss 1.0142318066067924, Correct 49, Time 0.10930585861206055 seconds
Epoch 270, Loss 1.1685530393453543, Correct 48, Time 0.10742640495300293 seconds
Epoch 280, Loss 0.917429560344019, Correct 48, Time 0.1069037914276123 seconds
Epoch 290, Loss 1.5878430090085356, Correct 48, Time 0.10889005661010742 seconds
Epoch 300, Loss 3.250316432323241, Correct 48, Time 0.1070408821105957 seconds
Epoch 310, Loss 2.2805614665503464, Correct 49, Time 0.10908102989196777 seconds
Epoch 320, Loss 1.477414891757925, Correct 48, Time 0.11295151710510254 seconds
Epoch 330, Loss 0.8149251814552207, Correct 49, Time 0.16804933547973633 seconds
Epoch 340, Loss 0.5373266771821735, Correct 49, Time 0.11397171020507812 seconds
Epoch 350, Loss 0.29982807514210524, Correct 49, Time 0.10855555534362793 seconds
Epoch 360, Loss 0.24292570343080516, Correct 50, Time 0.10848736763000488 seconds
Epoch 370, Loss 0.5802689924642933, Correct 50, Time 0.10782885551452637 seconds
Epoch 380, Loss 0.11957289169525157, Correct 48, Time 0.11038565635681152 seconds
Epoch 390, Loss 0.9487560787522403, Correct 48, Time 0.10862469673156738 seconds
Epoch 400, Loss 0.3590540472891047, Correct 49, Time 0.10872650146484375 seconds
Epoch 410, Loss 0.8938451136443433, Correct 49, Time 0.11968374252319336 seconds
Epoch 420, Loss 0.5484482082961325, Correct 49, Time 0.11132383346557617 seconds
Epoch 430, Loss 0.9175333953610019, Correct 48, Time 0.21046829223632812 seconds
Epoch 440, Loss 0.6102923969649474, Correct 48, Time 0.11140823364257812 seconds
Epoch 450, Loss 0.30821237295103393, Correct 48, Time 0.10830974578857422 seconds
Epoch 460, Loss 0.21374991640025437, Correct 48, Time 0.10860872268676758 seconds
Epoch 470, Loss 0.530832280197137, Correct 50, Time 0.10904359817504883 seconds
Epoch 480, Loss 0.27808911924563084, Correct 49, Time 0.10838723182678223 seconds
Epoch 490, Loss 0.3072473114850978, Correct 50, Time 0.10948848724365234 seconds
Average Time Per Epoch: 0.12288490182651068 s
```

Xor:
```
Epoch 0, Loss 6.330395483717392, Correct 32, Time 14.255102634429932 seconds
Epoch 10, Loss 4.522926814369849, Correct 38, Time 0.11731362342834473 seconds
Epoch 20, Loss 6.033851109850486, Correct 39, Time 0.10964202880859375 seconds
Epoch 30, Loss 5.2981015374684635, Correct 39, Time 0.10764813423156738 seconds
Epoch 40, Loss 3.565030783238192, Correct 42, Time 0.12564659118652344 seconds
Epoch 50, Loss 4.322137877115235, Correct 43, Time 0.10838651657104492 seconds
Epoch 60, Loss 3.144660140998094, Correct 44, Time 0.10946393013000488 seconds
Epoch 70, Loss 1.9136282591030316, Correct 44, Time 0.11581993103027344 seconds
Epoch 80, Loss 3.790563776897265, Correct 44, Time 0.10708498954772949 seconds
Epoch 90, Loss 3.4669194129731, Correct 45, Time 0.18593525886535645 seconds
Epoch 100, Loss 3.3795914357422565, Correct 45, Time 0.10857987403869629 seconds
Epoch 110, Loss 2.618916304006704, Correct 46, Time 0.11489033699035645 seconds
Epoch 120, Loss 3.760744888943395, Correct 46, Time 0.11970353126525879 seconds
Epoch 130, Loss 1.2841682855330592, Correct 47, Time 0.11089920997619629 seconds
Epoch 140, Loss 1.2517501950305414, Correct 48, Time 0.11299538612365723 seconds
Epoch 150, Loss 0.5061318962278344, Correct 50, Time 0.11050271987915039 seconds
Epoch 160, Loss 2.2243924033906857, Correct 50, Time 0.10675859451293945 seconds
Epoch 170, Loss 3.0457855680580685, Correct 47, Time 0.10787796974182129 seconds
Epoch 180, Loss 0.906038542095847, Correct 44, Time 0.10690641403198242 seconds
Epoch 190, Loss 1.1885709392256936, Correct 48, Time 0.21114873886108398 seconds
Epoch 200, Loss 1.2998819690547738, Correct 50, Time 0.10946226119995117 seconds
Epoch 210, Loss 3.092138526417445, Correct 48, Time 0.11265730857849121 seconds
Epoch 220, Loss 1.5107374091760575, Correct 49, Time 0.10872840881347656 seconds
Epoch 230, Loss 0.6783202842142986, Correct 49, Time 0.12560749053955078 seconds
Epoch 240, Loss 0.632316026905082, Correct 49, Time 0.10735869407653809 seconds
Epoch 250, Loss 0.8958843461775204, Correct 49, Time 0.1103665828704834 seconds
Epoch 260, Loss 0.37855194556974586, Correct 50, Time 0.11151528358459473 seconds
Epoch 270, Loss 0.8681709402072038, Correct 49, Time 0.1096656322479248 seconds
Epoch 280, Loss 1.1860059873742363, Correct 49, Time 0.10878849029541016 seconds
Epoch 290, Loss 0.4675737495974307, Correct 49, Time 0.23073291778564453 seconds
Epoch 300, Loss 0.5245558919021283, Correct 49, Time 0.10845422744750977 seconds
Epoch 310, Loss 0.2949532747366753, Correct 50, Time 0.1098785400390625 seconds
Epoch 320, Loss 1.328710515468395, Correct 49, Time 0.1100459098815918 seconds
Epoch 330, Loss 0.16886668064703475, Correct 50, Time 0.1082906723022461 seconds
Epoch 340, Loss 1.7359817233793504, Correct 48, Time 0.10859894752502441 seconds
Epoch 350, Loss 0.3135123014403213, Correct 49, Time 0.10877752304077148 seconds
Epoch 360, Loss 0.3543110389006435, Correct 49, Time 0.11185193061828613 seconds
Epoch 370, Loss 0.08025021144997783, Correct 50, Time 0.10779428482055664 seconds
Epoch 380, Loss 1.282671063499808, Correct 49, Time 0.10973668098449707 seconds
Epoch 390, Loss 0.4511280129746982, Correct 50, Time 0.2001054286956787 seconds
Epoch 400, Loss 0.8388394567501151, Correct 50, Time 0.1091458797454834 seconds
Epoch 410, Loss 1.2248139766243045, Correct 50, Time 0.10737156867980957 seconds
Epoch 420, Loss 0.10248782533802753, Correct 50, Time 0.1120138168334961 seconds
Epoch 430, Loss 0.8405103566014585, Correct 50, Time 0.11140108108520508 seconds
Epoch 440, Loss 0.0520065555431333, Correct 49, Time 0.1092064380645752 seconds
Epoch 450, Loss 0.2788104831396997, Correct 50, Time 0.1091153621673584 seconds
Epoch 460, Loss 1.1094304894596505, Correct 49, Time 0.11794233322143555 seconds
Epoch 470, Loss 1.0666833106130607, Correct 50, Time 0.10792422294616699 seconds
Epoch 480, Loss 0.1153103444881602, Correct 49, Time 0.10874223709106445 seconds
Epoch 490, Loss 1.394527870999093, Correct 49, Time 0.25347256660461426 seconds
Average Time Per Epoch: 0.12415481187059789 s
```