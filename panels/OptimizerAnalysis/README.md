### OptimizerAnalysis

The `OptimizerAnalysis` panel is used to explore results from an
Optimizer Search or Sweep. The [Comet Optimizer]() is used to
dynamically find the best set of hyperparameter values that will
minimize a Hyper Parameter Optimization tool (HPO) that can be used to
maximize a particular metric. The OptimizerAnalysis panel, combined
with the [Parallel Coordinate Chart](https://www.comet.com/docs/v2/guides/comet-ui/experiment-management/visualizations/parallel-coordinate-chart/)
allows detailed exploration of the results from your grid search or
sweep.


<table>
<tr>
<td>
<img src="https://raw.githubusercontent.com/comet-ml/comet-examples/refs/heads/master/panels/OptimizerAnalysis/optimizer-analysis.png" 
     style="max-width: 300px; max-height: 300px;">
</img>
</td>
</tr>
</table>

#### Python Panel

To include this panel from the github repo, use this code in a Custom Python Panel:

```
%include https://raw.githubusercontent.com/comet-ml/comet-examples/refs/heads/master/panels/OptimizerAnalysis/OptimizerAnalysis.py
```

Or, you can simply [copy the code](https://raw.githubusercontent.com/comet-ml/comet-examples/refs/heads/master/panels/OptimizerAnalysis/OptimizerAnalysis.py) into a custom Python Panel.

#### Resources

* Example Comet Project: [www.comet.com/examples/comet-example-optimizer](https://www.comet.com/examples/comet-example-optimizer/view/SA4f2JEsWKDzMaMLbW1yUYlc1/panels)
* [Optimizer Quickstart](https://www.comet.com/docs/v2/guides/optimizer/quickstart/)
* [Running Optimizer in Parallel](https://www.comet.com/docs/v2/guides/optimizer/run-in-parallel/#how-to-parallelize-comet-optimizer)
  * [Command-line](https://www.comet.com/docs/v2/api-and-sdk/command-line/reference/#comet-optimize)
* [Using 3rd-Party Optimizers](https://www.comet.com/docs/v2/guides/optimizer/third-party-optimizers/)
