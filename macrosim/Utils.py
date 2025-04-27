import pandas as pd
import numpy as np
from pysr import PySRRegressor, AbstractLoggerSpec, AbstractExpressionSpec

from sklearn.neighbors import LocalOutlierFactor

from typing import Literal, Callable, Optional
from dataclasses import dataclass, asdict


class DataUtils:
    """Static Class for data transformation utilities"""

    @staticmethod
    def lof_filter(series: pd.Series | pd.DataFrame, n_neighbors: Optional[int] = None) -> pd.Series:
        """
        Local Outlier Factor (LOF) based outlier removal util. Returns the unfiltered series if LOF removes more than
        25% of the data.
        :param series: pd.Series object or a single column pd.DataFrame
        :param n_neighbors: Size of neighborhood to use for outlier removal
        :return: filtered pd.Series object
        """
        if isinstance(series, pd.Series):
            series = series.to_frame()
            
        n = n_neighbors or int(np.floor(np.sqrt(len(series))))
        lof = LocalOutlierFactor(n_neighbors=n, contamination='auto')
        lof_mask = np.where(lof.fit_predict(series) == -1, True, False)

        filtered = series[lof_mask]
        return filtered if len(filtered) >= .75 * len(series) else series

    @staticmethod
    def ma_smoothing(series: pd.Series, window: Optional[int] = None) -> pd.Series:
        """
        Moving average smoothing for data generalization.
        :param series: pd.Series object
        :param window: Window size for moving average
        :return: smoothed pd.Series object
        """

        window = window or int(np.floor(np.sqrt(len(series))))
        return series.rolling(window).mean()

    @staticmethod
    def ewma_smoothing(series: pd.Series, span: Optional[int] = None) -> pd.Series:
        """
        Exponential moving average smoothing for data generalization.
        :param series: pd.Series object
        :param span: Span size for moving average
        :return: smoothed pd.Series object
        """

        span = span or int(np.floor(np.sqrt(len(series))))
        return series.ewm(span=span).mean()

    @staticmethod
    def get_lags(series: pd.Series, n_lags: Optional[int] = None) -> pd.DataFrame:
        """
        Creates a lagged feature space from a series.
        :param series: pd.Series object
        :param n_lags: Number of lags to create
        :return: Lagged feature as pd.DataFrame
        """

        n_lags = int(np.ceil(len(series)**(1/3)))
        lagged_df = series.to_frame()
        lagged_df.columns = ['X_t']

        for lag in range(1, n_lags+1):
            lagged_df[f"X_t{lag}"] = lagged_df['X_t'].shift(lag)

        lagged_df = lagged_df.dropna(how='any')
        return lagged_df


@dataclass(frozen=True)
class SrConfig:
    """
    Kwargs container for PySRRegressor instances.

    Attributes
    ----------
    model_selection : str
        Model selection criterion when selecting a final expression from
        the list of best expression at each complexity.
        Can be `'accuracy'`, `'best'`, or `'score'`. Default is `'best'`.
        `'accuracy'` selects the candidate model with the lowest loss
        (highest accuracy).
        `'score'` selects the candidate model with the highest score.
        Score is defined as the negated derivative of the log-loss with
        respect to complexity - if an expression has a much better
        loss at a slightly higher complexity, it is preferred.
        `'best'` selects the candidate model with the highest score
        among expressions with a loss better than at least 1.5x the
        most accurate model.
    binary_operators : list[str]
        List of strings for binary operators used in the search.
        See the [operators page](https://ai.damtp.cam.ac.uk/pysr/operators/)
        for more details.
        Default is `["+", "-", "*", "/"]`.
    unary_operators : list[str]
        Operators which only take a single scalar as input.
        For example, `"cos"` or `"exp"`.
        Default is `None`.
    expression_spec : AbstractExpressionSpec
        The type of expression to search for. By default,
        this is just `ExpressionSpec()`. You can also use
        `TemplateExpressionSpec(...)` which allows you to specify
        a custom template for the expressions.
        Default is `ExpressionSpec()`.
    niterations : int
        Number of iterations of the algorithm to run. The best
        equations are printed and migrate between populations at the
        end of each iteration.
        Default is `100`.
    populations : int
        Number of populations running.
        Default is `31`.
    population_size : int
        Number of individuals in each population.
        Default is `27`.
    max_evals : int
        Limits the total number of evaluations of expressions to
        this number.  Default is `None`.
    maxsize : int
        Max complexity of an equation.  Default is `30`.
    maxdepth : int
        Max depth of an equation. You can use both `maxsize` and
        `maxdepth`. `maxdepth` is by default not used.
        Default is `None`.
    warmup_maxsize_by : float
        Whether to slowly increase max size from a small number up to
        the maxsize (if greater than 0).  If greater than 0, says the
        fraction of training time at which the current maxsize will
        reach the user-passed maxsize.
        Default is `0.0`.
    timeout_in_seconds : float
        Make the search return early once this many seconds have passed.
        Default is `None`.
    constraints : dict[str, int | tuple[int,int]]
        Dictionary of int (unary) or 2-tuples (binary), this enforces
        maxsize constraints on the individual arguments of operators.
        E.g., `'pow': (-1, 1)` says that power laws can have any
        complexity left argument, but only 1 complexity in the right
        argument. Use this to force more interpretable solutions.
        Default is `None`.
    nested_constraints : dict[str, dict]
        Specifies how many times a combination of operators can be
        nested. For example, `{"sin": {"cos": 0}}, "cos": {"cos": 2}}`
        specifies that `cos` may never appear within a `sin`, but `sin`
        can be nested with itself an unlimited number of times. The
        second term specifies that `cos` can be nested up to 2 times
        within a `cos`, so that `cos(cos(cos(x)))` is allowed
        (as well as any combination of `+` or `-` within it), but
        `cos(cos(cos(cos(x))))` is not allowed. When an operator is not
        specified, it is assumed that it can be nested an unlimited
        number of times. This requires that there is no operator which
        is used both in the unary operators and the binary operators
        (e.g., `-` could be both subtract, and negation). For binary
        operators, you only need to provide a single number: both
        arguments are treated the same way, and the max of each
        argument is constrained.
        Default is `None`.
    elementwise_loss : str
        String of Julia code specifying an elementwise loss function.
        Can either be a loss from LossFunctions.jl, or your own loss
        written as a function. Examples of custom written losses include:
        `myloss(x, y) = abs(x-y)` for non-weighted, or
        `myloss(x, y, w) = w*abs(x-y)` for weighted.
        The included losses include:
        Regression: `LPDistLoss{P}()`, `L1DistLoss()`,
        `L2DistLoss()` (mean square), `LogitDistLoss()`,
        `HuberLoss(d)`, `L1EpsilonInsLoss(ϵ)`, `L2EpsilonInsLoss(ϵ)`,
        `PeriodicLoss(c)`, `QuantileLoss(τ)`.
        Classification: `ZeroOneLoss()`, `PerceptronLoss()`,
        `L1HingeLoss()`, `SmoothedL1HingeLoss(γ)`,
        `ModifiedHuberLoss()`, `L2MarginLoss()`, `ExpLoss()`,
        `SigmoidLoss()`, `DWDMarginLoss(q)`.
        Default is `"L2DistLoss()"`.
    loss_function : str
        Alternatively, you can specify the full objective function as
        a snippet of Julia code, including any sort of custom evaluation
        (including symbolic manipulations beforehand), and any sort
        of loss function or regularizations. The default `loss_function`
        used in SymbolicRegression.jl is roughly equal to:
        ```julia
        function eval_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
            prediction, flag = eval_tree_array(tree, dataset.X, options)
            if !flag
                return L(Inf)
            end
            return sum((prediction .- dataset.y) .^ 2) / dataset.n
        end
        ```
        where the example elementwise loss is mean-squared error.
        You may pass a function with the same arguments as this (note
        that the name of the function doesn't matter). Here,
        both `prediction` and `dataset.y` are 1D arrays of length `dataset.n`.
        Default is `None`.
    loss_function_expression : str
        Similar to `loss_function`, but takes as input the full
        expression object as the first argument, rather than
        the innermost `AbstractExpressionNode`. This is useful
        for specifying custom loss functions on `TemplateExpressionSpec`.
        Default is `None`.
    complexity_of_operators : dict[str, int | float]
        If you would like to use a complexity other than 1 for an
        operator, specify the complexity here. For example,
        `{"sin": 2, "+": 1}` would give a complexity of 2 for each use
        of the `sin` operator, and a complexity of 1 for each use of
        the `+` operator (which is the default). You may specify real
        numbers for a complexity, and the total complexity of a tree
        will be rounded to the nearest integer after computing.
        Default is `None`.
    complexity_of_constants : int | float
        Complexity of constants. Default is `1`.
    complexity_of_variables : int | float | list[int | float]
        Global complexity of variables. To set different complexities for
        different variables, pass a list of complexities to the `fit` method
        with keyword `complexity_of_variables`. You cannot use both.
        Default is `1`.
    complexity_mapping : str
        Alternatively, you can pass a function (a string of Julia code) that
        takes the expression as input and returns the complexity. Make sure that
        this operates on `AbstractExpression` (and unpacks to `AbstractExpressionNode`),
        and returns an integer.
        Default is `None`.
    parsimony : float
        Multiplicative factor for how much to punish complexity.
        Default is `0.0`.
    dimensional_constraint_penalty : float
        Additive penalty for if dimensional analysis of an expression fails.
        By default, this is `1000.0`.
    dimensionless_constants_only : bool
        Whether to only search for dimensionless constants, if using units.
        Default is `False`.
    use_frequency : bool
        Whether to measure the frequency of complexities, and use that
        instead of parsimony to explore equation space. Will naturally
        find equations of all complexities.
        Default is `True`.
    use_frequency_in_tournament : bool
        Whether to use the frequency mentioned above in the tournament,
        rather than just the simulated annealing.
        Default is `True`.
    adaptive_parsimony_scaling : float
        If the adaptive parsimony strategy (`use_frequency` and
        `use_frequency_in_tournament`), this is how much to (exponentially)
        weight the contribution. If you find that the search is only optimizing
        the most complex expressions while the simpler expressions remain stagnant,
        you should increase this value.
        Default is `1040.0`.
    alpha : float
        Initial temperature for simulated annealing
        (requires `annealing` to be `True`).
        Default is `3.17`.
    annealing : bool
        Whether to use annealing.  Default is `False`.
    early_stop_condition : float | str
        Stop the search early if this loss is reached. You may also
        pass a string containing a Julia function which
        takes a loss and complexity as input, for example:
        `"f(loss, complexity) = (loss < 0.1) && (complexity < 10)"`.
        Default is `None`.
    ncycles_per_iteration : int
        Number of total mutations to run, per 10 samples of the
        population, per iteration.
        Default is `380`.
    fraction_replaced : float
        How much of population to replace with migrating equations from
        other populations.
        Default is `0.00036`.
    fraction_replaced_hof : float
        How much of population to replace with migrating equations from
        hall of fame. Default is `0.0614`.
    weight_add_node : float
        Relative likelihood for mutation to add a node.
        Default is `2.47`.
    weight_insert_node : float
        Relative likelihood for mutation to insert a node.
        Default is `0.0112`.
    weight_delete_node : float
        Relative likelihood for mutation to delete a node.
        Default is `0.870`.
    weight_do_nothing : float
        Relative likelihood for mutation to leave the individual.
        Default is `0.273`.
    weight_mutate_constant : float
        Relative likelihood for mutation to change the constant slightly
        in a random direction.
        Default is `0.0346`.
    weight_mutate_operator : float
        Relative likelihood for mutation to swap an operator.
        Default is `0.293`.
    weight_swap_operands : float
        Relative likehood for swapping operands in binary operators.
        Default is `0.198`.
    weight_rotate_tree : float
        How often to perform a tree rotation at a random node.
        Default is `4.26`.
    weight_randomize : float
        Relative likelihood for mutation to completely delete and then
        randomly generate the equation
        Default is `0.000502`.
    weight_simplify : float
        Relative likelihood for mutation to simplify constant parts by evaluation
        Default is `0.00209`.
    weight_optimize: float
        Constant optimization can also be performed as a mutation, in addition to
        the normal strategy controlled by `optimize_probability` which happens
        every iteration. Using it as a mutation is useful if you want to use
        a large `ncycles_periteration`, and may not optimize very often.
        Default is `0.0`.
    crossover_probability : float
        Absolute probability of crossover-type genetic operation, instead of a mutation.
        Default is `0.0259`.
    skip_mutation_failures : bool
        Whether to skip mutation and crossover failures, rather than
        simply re-sampling the current member.
        Default is `True`.
    migration : bool
        Whether to migrate.  Default is `True`.
    hof_migration : bool
        Whether to have the hall of fame migrate.  Default is `True`.
    topn : int
        How many top individuals migrate from each population.
        Default is `12`.
    should_simplify : bool
        Whether to use algebraic simplification in the search. Note that only
        a few simple rules are implemented. Default is `True`.
    should_optimize_constants : bool
        Whether to numerically optimize constants (Nelder-Mead/Newton)
        at the end of each iteration. Default is `True`.
    optimizer_algorithm : str
        Optimization scheme to use for optimizing constants. Can currently
        be `NelderMead` or `BFGS`.
        Default is `"BFGS"`.
    optimizer_nrestarts : int
        Number of time to restart the constants optimization process with
        different initial conditions.
        Default is `2`.
    optimizer_f_calls_limit : int
        How many function calls to allow during optimization.
        Default is `10_000`.
    optimize_probability : float
        Probability of optimizing the constants during a single iteration of
        the evolutionary algorithm.
        Default is `0.14`.
    optimizer_iterations : int
        Number of iterations that the constants optimizer can take.
        Default is `8`.
    perturbation_factor : float
        Constants are perturbed by a max factor of
        (perturbation_factor*T + 1). Either multiplied by this or
        divided by this.
        Default is `0.129`.
    probability_negate_constant : float
        Probability of negating a constant in the equation when mutating it.
        Default is `0.00743`.
    tournament_selection_n : int
        Number of expressions to consider in each tournament.
        Default is `15`.
    tournament_selection_p : float
        Probability of selecting the best expression in each
        tournament. The probability will decay as p*(1-p)^n for other
        expressions, sorted by loss.
        Default is `0.982`.
    parallelism: Literal["serial", "multithreading", "multiprocessing"] | None
        Parallelism to use for the search. Can be `"serial"`, `"multithreading"`, or `"multiprocessing"`.
        Default is `"multithreading"`.
    procs: int | None
        Number of processes to use for parallelism. If `None`, defaults to `cpu_count()`.
        Default is `None`.
    cluster_manager : str
        For distributed computing, this sets the job queue system. Set
        to one of "slurm", "pbs", "lsf", "sge", "qrsh", "scyld", or
        "htc". If set to one of these, PySR will run in distributed
        mode, and use `procs` to figure out how many processes to launch.
        Default is `None`.
    heap_size_hint_in_bytes : int
        For multiprocessing, this sets the `--heap-size-hint` parameter
        for new Julia processes. This can be configured when using
        multi-node distributed compute, to give a hint to each process
        about how much memory they can use before aggressive garbage
        collection.
    batching : bool
        Whether to compare population members on small batches during
        evolution. Still uses full dataset for comparing against hall
        of fame. Default is `False`.
    batch_size : int
        The amount of data to use if doing batching. Default is `50`.
    fast_cycle : bool
        Batch over population subsamples. This is a slightly different
        algorithm than regularized evolution, but does cycles 15%
        faster. May be algorithmically less efficient.
        Default is `False`.
    turbo: bool
        (Experimental) Whether to use LoopVectorization.jl to speed up the
        search evaluation. Certain operators may not be supported.
        Does not support 16-bit precision floats.
        Default is `False`.
    bumper: bool
        (Experimental) Whether to use Bumper.jl to speed up the search
        evaluation. Does not support 16-bit precision floats.
        Default is `False`.
    precision : int
        What precision to use for the data. By default this is `32`
        (float32), but you can select `64` or `16` as well, giving
        you 64 or 16 bits of floating point precision, respectively.
        If you pass complex data, the corresponding complex precision
        will be used (i.e., `64` for complex128, `32` for complex64).
        Default is `32`.
    autodiff_backend : Literal["Zygote"] | None
        Which backend to use for automatic differentiation during constant
        optimization. Currently only `"Zygote"` is supported. The default,
        `None`, uses forward-mode or finite difference.
        Default is `None`.
    random_state : int, Numpy RandomState instance or None
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.
        Default is `None`.
    deterministic : bool
        Make a PySR search give the same result every run.
        To use this, you must turn off parallelism
        (with `parallelism="serial"`),
        and set `random_state` to a fixed seed.
        Default is `False`.
    warm_start : bool
        Tells fit to continue from where the last call to fit finished.
        If false, each call to fit will be fresh, overwriting previous results.
        Default is `False`.
    verbosity : int
        What verbosity level to use. 0 means minimal print statements.
        Default is `1`.
    update_verbosity : int
        What verbosity level to use for package updates.
        Will take value of `verbosity` if not given.
        Default is `None`.
    print_precision : int
        How many significant digits to print for floats. Default is `5`.
    progress : bool
        Whether to use a progress bar instead of printing to stdout.
        Default is `True`.
    logger_spec: AbstractLoggerSpec | None
        Logger specification for the Julia backend. See, for example,
        `TensorBoardLoggerSpec`.
        Default is `None`.
    input_stream : str
        The stream to read user input from. By default, this is `"stdin"`.
        If you encounter issues with reading from `stdin`, like a hang,
        you can simply pass `"devnull"` to this argument. You can also
        reference an arbitrary Julia object in the `Main` namespace.
        Default is `"stdin"`.
    run_id : str
        A unique identifier for the run. Will be generated using the
        current date and time if not provided.
        Default is `None`.
    output_directory : str
        The base directory to save output files to. Files
        will be saved in a subdirectory according to the run ID.
        Will be set to `outputs/` if not provided.
        Default is `None`.
    temp_equation_file : bool
        Whether to put the hall of fame file in the temp directory.
        Deletion is then controlled with the `delete_tempfiles`
        parameter.
        Default is `False`.
    tempdir : str
        directory for the temporary files. Default is `None`.
    delete_tempfiles : bool
        Whether to delete the temporary files after finishing.
        Default is `True`.
    update: bool
        Whether to automatically update Julia packages when `fit` is called.
        You should make sure that PySR is up-to-date itself first, as
        the packaged Julia packages may not necessarily include all
        updated dependencies.
        Default is `False`.
    output_jax_format : bool
        Whether to create a 'jax_format' column in the output,
        containing jax-callable functions and the default parameters in
        a jax array.
        Default is `False`.
    output_torch_format : bool
        Whether to create a 'torch_format' column in the output,
        containing a torch module with trainable parameters.
        Default is `False`.
    extra_sympy_mappings : dict[str, Callable]
        Provides mappings between custom `binary_operators` or
        `unary_operators` defined in julia strings, to those same
        operators defined in sympy.
        E.G if `unary_operators=["inv(x)=1/x"]`, then for the fitted
        model to be export to sympy, `extra_sympy_mappings`
        would be `{"inv": lambda x: 1/x}`.
        Default is `None`.
    extra_jax_mappings : dict[Callable, str]
        Similar to `extra_sympy_mappings` but for model export
        to jax. The dictionary maps sympy functions to jax functions.
        For example: `extra_jax_mappings={sympy.sin: "jnp.sin"}` maps
        the `sympy.sin` function to the equivalent jax expression `jnp.sin`.
        Default is `None`.
    extra_torch_mappings : dict[Callable, Callable]
        The same as `extra_jax_mappings` but for model export
        to pytorch. Note that the dictionary keys should be callable
        pytorch expressions.
        For example: `extra_torch_mappings={sympy.sin: torch.sin}`.
        Default is `None`.
    denoise : bool
        Whether to use a Gaussian Process to denoise the data before
        inputting to PySR. Can help PySR fit noisy data.
        Default is `False`.
    select_k_features : int
        Whether to run feature selection in Python using random forests,
        before passing to the symbolic regression code. None means no
        feature selection; an int means select that many features.
        Default is `None`.
    """

    model_selection: Literal["best", "accuracy", "score"] = "best"
    binary_operators: list[str] | None = None
    unary_operators: list[str] | None = None
    expression_spec: AbstractExpressionSpec | None = None
    niterations: int = 100
    populations: int = 31
    population_size: int = 27
    max_evals: int | None = None
    maxsize: int = 30
    maxdepth: int | None = None
    warmup_maxsize_by: float | None = None
    timeout_in_seconds: float | None = None
    constraints: dict[str, int | tuple[int, int]] | None = None
    nested_constraints: dict[str, dict[str, int]] | None = None
    elementwise_loss: str | None = None
    loss_function: str | None = None
    loss_function_expression: str | None = None
    complexity_of_operators: dict[str, int | float] | None = None
    complexity_of_constants: int | float | None = None
    complexity_of_variables: int | float | list[int | float] | None = None
    complexity_mapping: str | None = None
    parsimony: float = 0.0
    dimensional_constraint_penalty: float | None = None
    dimensionless_constants_only: bool = False
    use_frequency: bool = True
    use_frequency_in_tournament: bool = True
    adaptive_parsimony_scaling: float = 1040.0
    alpha: float = 3.17
    annealing: bool = False
    early_stop_condition: float | str | None = None
    ncycles_per_iteration: int = 380
    fraction_replaced: float = 0.00036
    fraction_replaced_hof: float = 0.0614
    weight_add_node: float = 2.47
    weight_insert_node: float = 0.0112
    weight_delete_node: float = 0.870
    weight_do_nothing: float = 0.273
    weight_mutate_constant: float = 0.0346
    weight_mutate_operator: float = 0.293
    weight_swap_operands: float = 0.198
    weight_rotate_tree: float = 4.26
    weight_randomize: float = 0.000502
    weight_simplify: float = 0.00209
    weight_optimize: float = 0.0
    crossover_probability: float = 0.0259
    skip_mutation_failures: bool = True
    migration: bool = True
    hof_migration: bool = True
    topn: int = 12
    should_simplify: bool = True
    should_optimize_constants: bool = True
    optimizer_algorithm: Literal["BFGS", "NelderMead"] = "BFGS"
    optimizer_nrestarts: int = 2
    optimizer_f_calls_limit: int | None = None
    optimize_probability: float = 0.14
    optimizer_iterations: int = 8
    perturbation_factor: float = 0.129
    probability_negate_constant: float = 0.00743
    tournament_selection_n: int = 15
    tournament_selection_p: float = 0.982
    parallelism: Literal["serial", "multithreading", "multiprocessing"] | None = None
    procs: int | None = None
    cluster_manager: Literal["slurm", "pbs", "lsf", "sge", "qrsh", "scyld", "htc"] | None = None
    heap_size_hint_in_bytes: int | None = None
    batching: bool = False
    batch_size: int = 50
    fast_cycle: bool = False
    turbo: bool = False
    bumper: bool = False
    precision: Literal[16, 32, 64] = 32
    autodiff_backend: Literal["Zygote"] | None = None
    random_state: int | np.random.RandomState | None = None
    deterministic: bool = False
    warm_start: bool = False
    verbosity: int = 1
    update_verbosity: int | None = None
    print_precision: int = 5
    progress: bool = True
    logger_spec: AbstractLoggerSpec | None = None
    input_stream: str = "stdin"
    run_id: str | None = None
    output_directory: str | None = None
    temp_equation_file: bool = False
    tempdir: str | None = None
    delete_tempfiles: bool = True
    update: bool = False
    output_jax_format: bool = False
    output_torch_format: bool = False
    extra_sympy_mappings: dict[str, Callable] | None = None
    extra_torch_mappings: dict[Callable, Callable] | None = None
    extra_jax_mappings: dict[Callable, str] | None = None
    denoise: bool = False
    select_k_features: int | None = None


def sr_generator(config: SrConfig) -> PySRRegressor:
    """
    Generates a PySRRegressor object with the specified keyword arguments.
    :return: PySRRegressor object
    """
    return PySRRegressor(**asdict(config))


