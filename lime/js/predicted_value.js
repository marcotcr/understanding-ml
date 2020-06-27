import d3 from 'd3';
import {range, sortBy} from 'lodash';

class PredictedValue {
  // svg: d3 object with the svg in question
  // class_names: array of class names
  // predict_probas: array of prediction probabilities
  constructor(svg, predicted_value, min_value, max_value, title='Predicted value', log_coords = false) {

    if (min_value == max_value){
        var width_proportion = 1.0;
    } else {
        var width_proportion = (predicted_value - min_value) / (max_value - min_value);
    }


    let width = parseInt(svg.style('width'))

    this.color = d3.scale.category10()
    this.color('predicted_value')
    // + 2 is due to it being a float
    let num_digits = Math.floor(Math.max(Math.log10(Math.abs(min_value)), Math.log10(Math.abs(max_value)))) + 2
    num_digits = Math.max(num_digits, 3)

    let corner_width = 12 * num_digits;
    let corner_padding = 5.5 * num_digits;
    let bar_x = corner_width + corner_padding;
    let bar_width = width - corner_width * 2 - corner_padding * 2;
    let x_scale = d3.scale.linear().range([0, bar_width]);
    let bar_height = 17;
    let bar_yshift= title === '' ? 0 : 35;
    let n_bars = 1;
    let this_object = this;
    if (title !== '') {
      svg.append('text')
        .text(title)
        .attr('x', 20)
        .attr('y', 20);
    }
    let bar_y = bar_yshift;
    let bar = svg.append("g");

  //filled in bar representing predicted value in range
  let rect = bar.append("rect");
  rect.attr("x", bar_x)
      .attr("y", bar_y)
      .attr("height", bar_height)
      .attr("width", x_scale(width_proportion))
      .style("fill", this.color);

  //empty box representing range
  bar.append("rect").attr("x", bar_x)
      .attr("y", bar_y)
      .attr("height", bar_height)
      .attr("width",x_scale(1))
      .attr("fill-opacity", 0)
      .attr("stroke", "black");
  let text = bar.append("text");
  text.classed("prob_text", true);
  text.attr("y", bar_y + bar_height - 3).attr("fill", "black").style("font", "14px tahoma, sans-serif");


  //text for min value
  text = bar.append("text");
  text.attr("x", bar_x - corner_padding)
      .attr("y", bar_y + bar_height - 3)
      .attr("fill", "black")
      .attr("text-anchor", "end")
      .style("font", "14px tahoma, sans-serif")
      .text(min_value.toFixed(2));

  //text for range min annotation
  let v_adjust_min_value_annotation = text.node().getBBox().height;
  text = bar.append("text");
  text.attr("x", bar_x - corner_padding)
      .attr("y", bar_y + bar_height - 3 + v_adjust_min_value_annotation)
      .attr("fill", "black")
      .attr("text-anchor", "end")
      .style("font", "14px tahoma, sans-serif")
      .text("(min)");


  //text for predicted value
  // console.log('bar height: ' + bar_height)
  text = bar.append("text");
  text.text(predicted_value.toFixed(2));
  // let h_adjust_predicted_value_text = text.node().getBBox().width / 2;
  let v_adjust_predicted_value_text = text.node().getBBox().height;
  text.attr("x", bar_x + x_scale(width_proportion))
      .attr("y", bar_y + bar_height + v_adjust_predicted_value_text)
      .attr("fill", "black")
      .attr("text-anchor", "middle")
      .style("font", "14px tahoma, sans-serif")





  //text for max value
  text = bar.append("text");
  text.text(max_value.toFixed(2));
  // let h_adjust = text.node().getBBox().width;
  text.attr("x", bar_x + bar_width + corner_padding)
      .attr("y", bar_y + bar_height - 3)
      .attr("fill", "black")
      .attr("text-anchor", "begin")
      .style("font", "14px tahoma, sans-serif");


  //text for range max annotation
  let v_adjust_max_value_annotation = text.node().getBBox().height;
  text = bar.append("text");
  text.attr("x", bar_x + bar_width + corner_padding)
      .attr("y", bar_y + bar_height - 3 + v_adjust_min_value_annotation)
      .attr("fill", "black")
      .attr("text-anchor", "begin")
      .style("font", "14px tahoma, sans-serif")
      .text("(max)");


  //readjust svg size
  // let svg_width = width + 1 * h_adjust;
  // svg.style('width', svg_width + 'px');

  this.svg_height = n_bars * (bar_height) + bar_yshift + (2 * text.node().getBBox().height) + 10;
  svg.style('height', this.svg_height + 'px');
  if (log_coords) {
      console.log("svg width: " + svg_width);
      console.log("svg height: " + this.svg_height);
      console.log("bar_y: " + bar_y);
      console.log("bar_x: " + bar_x);
      console.log("Min value: " + min_value);
      console.log("Max value: " + max_value);
      console.log("Pred value: " + predicted_value);
     }
  }
}


export default PredictedValue;
