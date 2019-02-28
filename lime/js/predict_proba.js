import d3 from "d3";
import { range, sortBy } from "lodash";

class PredictProba {
  // svg: d3 object with the svg in question
  // class_names: array of class names
  // predict_probas: array of prediction probabilities
  constructor(
    svg,
    class_names,
    predict_probas,
    title = "Prediction probabilities"
  ) {
    let width = parseInt(svg.style("width"));
    this.names = class_names;
    this.names.push("Other");
    if (class_names.length < 10) {
      this.colors = d3.scale.category10().domain(this.names);
      this.colors_i = d3.scale.category10().domain(range(this.names.length));
    } else {
      this.colors = d3.scale.category20().domain(this.names);
      this.colors_i = d3.scale.category20().domain(range(this.names.length));
    }
    let [names, data] = this.map_classes(this.names, predict_probas);
    let bar_x = width - 125;
    let class_names_width = bar_x;
    let bar_width = width - bar_x - 32;
    let x_scale = d3.scale.linear().range([0, bar_width]);
    let bar_height = 17;
    let space_between_bars = 5;
    let bar_yshift = title === "" ? 0 : 35;
    let n_bars = Math.min(5, data.length);
    this.svg_height = n_bars * (bar_height + space_between_bars) + bar_yshift;
    svg.style("height", this.svg_height + "px");
    let this_object = this;
    if (title !== "") {
      svg
        .append("text")
        .text(title)
        .attr("x", 20)
        .attr("y", 20);
    }
    let bar_y = i => (bar_height + space_between_bars) * i + bar_yshift;
    let bar = svg.append("g");

    for (let i of range(data.length)) {
      var color = this.colors(names[i]);
      if (names[i] == "Other" && this.names.length > 20) {
        color = "#5F9EA0";
      }
      let rect = bar.append("rect");
      rect
        .attr("x", bar_x)
        .attr("y", bar_y(i))
        .attr("height", bar_height)
        .attr("width", x_scale(data[i]))
        .style("fill", color);
      bar
        .append("rect")
        .attr("x", bar_x)
        .attr("y", bar_y(i))
        .attr("height", bar_height)
        .attr("width", bar_width - 1)
        .attr("fill-opacity", 0)
        .attr("stroke", "black");
      let text = bar.append("text");
      text.classed("prob_text", true);
      text
        .attr("y", bar_y(i) + bar_height - 3)
        .attr("fill", "black")
        .style("font", "14px tahoma, sans-serif");
      text = bar.append("text");
      text
        .attr("x", bar_x + x_scale(data[i]) + 5)
        .attr("y", bar_y(i) + bar_height - 3)
        .attr("fill", "black")
        .style("font", "14px tahoma, sans-serif")
        .text(data[i].toFixed(2));
      text = bar.append("text");
      text
        .attr("x", bar_x - 10)
        .attr("y", bar_y(i) + bar_height - 3)
        .attr("fill", "black")
        .attr("text-anchor", "end")
        .style("font", "14px tahoma, sans-serif")
        .text(names[i]);
      while (text.node().getBBox()["width"] + 1 > class_names_width - 10) {
        // TODO: ta mostrando só dois, e talvez quando hover mostrar o texto
        // todo
        let cur_text = text.text().slice(0, text.text().length - 5);
        text.text(cur_text + "...");
        if (cur_text === "") {
          break;
        }
      }
    }
  }
  map_classes(class_names, predict_proba) {
    if (class_names.length <= 6) {
      return [class_names, predict_proba];
    }
    let class_dict = range(predict_proba.length).map(i => ({
      name: class_names[i],
      prob: predict_proba[i],
      i: i
    }));
    let sorted = sortBy(class_dict, d => -d.prob);
    let other = new Set();
    range(4, sorted.length).map(d => other.add(sorted[d].name));
    let other_prob = 0;
    let ret_probs = [];
    let ret_names = [];
    for (let d of range(sorted.length)) {
      if (other.has(sorted[d].name)) {
        other_prob += sorted[d].prob;
      } else {
        ret_probs.push(sorted[d].prob);
        ret_names.push(sorted[d].name);
      }
    }
    ret_names.push("Other");
    ret_probs.push(other_prob);
    return [ret_names, ret_probs];
  }
}
export default PredictProba;
