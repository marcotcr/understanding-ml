import d3 from "d3";
import Barchart from "./bar_chart.js";
import { range, sortBy } from "lodash";
class Explanation {
  constructor(class_names) {
    this.names = class_names;
    if (class_names.length < 10) {
      this.colors = d3.scale.category10().domain(this.names);
      this.colors_i = d3.scale.category10().domain(range(this.names.length));
    } else {
      this.colors = d3.scale.category20().domain(this.names);
      this.colors_i = d3.scale.category20().domain(range(this.names.length));
    }
  }
  // exp: [(feature-name, weight), ...]
  // label: int
  // div: d3 selection
  show(exp, label, div) {
    let svg = div.append("svg").style("width", "100%");
    let colors = ["#5F9EA0", this.colors_i(label)];
    let names = [`NOT ${this.names[label]}`, this.names[label]];
    if (this.names.length == 2) {
      colors = [this.colors_i(0), this.colors_i(1)];
      names = this.names;
    }
    let plot = new Barchart(svg, exp, true, names, colors, true, 10);
    svg.style("height", plot.svg_height);
  }
  // exp has all ocurrences of words, with start index and weight:
  // exp = [('word', 132, -0.13), ('word3', 111, 1.3)
  show_raw_text(exp, label, raw, div, opacity = true) {
    //let colors=['#5F9EA0', this.colors(this.exp['class'])];
    let colors = ["#5F9EA0", this.colors_i(label)];
    if (this.names.length == 2) {
      colors = [this.colors_i(0), this.colors_i(1)];
    }
    let word_lists = [[], []];
    let max_weight = -1;
    for (let [word, start, weight] of exp) {
      if (weight > 0) {
        word_lists[1].push([start, start + word.length, weight]);
      } else {
        word_lists[0].push([start, start + word.length, -weight]);
      }
      max_weight = Math.max(max_weight, Math.abs(weight));
    }
    if (!opacity) {
      max_weight = 0;
    }
    this.display_raw_text(div, raw, word_lists, colors, max_weight, true);
  }
  // exp is list of (feature_name, value, weight)
  show_raw_tabular(exp, label, div) {
    div.classed("lime", true).classed("table_div", true);
    let colors = ["#5F9EA0", this.colors_i(label)];
    if (this.names.length == 2) {
      colors = [this.colors_i(0), this.colors_i(1)];
    }
    const table = div.append("table");
    const thead = table.append("tr");
    thead.append("td").text("Feature");
    thead.append("td").text("Value");
    thead.style("color", "black").style("font-size", "20px");
    for (let [fname, value, weight] of exp) {
      const tr = table.append("tr");
      tr.style("border-style", "hidden");
      tr.append("td").text(fname);
      tr.append("td").text(value);
      if (weight > 0) {
        tr.style("background-color", colors[1]);
      } else if (weight < 0) {
        tr.style("background-color", colors[0]);
      } else {
        tr.style("color", "black");
      }
    }
  }
  hexToRgb(hex) {
    let result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result
      ? {
          r: parseInt(result[1], 16),
          g: parseInt(result[2], 16),
          b: parseInt(result[3], 16)
        }
      : null;
  }
  applyAlpha(hex, alpha) {
    let components = this.hexToRgb(hex);
    return (
      "rgba(" +
      components.r +
      "," +
      components.g +
      "," +
      components.b +
      "," +
      alpha.toFixed(3) +
      ")"
    );
  }
  // sord_lists is an array of arrays, of length (colors). if with_positions is true,
  // word_lists is an array of [start,end] positions instead
  display_raw_text(
    div,
    raw_text,
    word_lists = [],
    colors = [],
    max_weight = 1,
    positions = false
  ) {
    div.classed("lime", true).classed("text_div", true);
    div.append("h3").text("Text with highlighted words");
    let highlight_tag = "span";
    let text_span = div
      .append("span")
      .style("white-space", "pre-wrap")
      .text(raw_text);
    let position_lists = word_lists;
    if (!positions) {
      position_lists = this.wordlists_to_positions(word_lists, raw_text);
    }
    let objects = [];
    for (let i of range(position_lists.length)) {
      position_lists[i].map(x =>
        objects.push({
          label: i,
          start: x[0],
          end: x[1],
          alpha: max_weight === 0 ? 1 : x[2] / max_weight
        })
      );
    }
    objects = sortBy(objects, x => x["start"]);
    let node = text_span.node().childNodes[0];
    let subtract = 0;
    for (let obj of objects) {
      let word = raw_text.slice(obj.start, obj.end);
      let start = obj.start - subtract;
      let end = obj.end - subtract;
      let match = document.createElement(highlight_tag);
      match.appendChild(document.createTextNode(word));
      match.style.backgroundColor = this.applyAlpha(
        colors[obj.label],
        obj.alpha
      );
      let after = node.splitText(start);
      after.nodeValue = after.nodeValue.substring(word.length);
      node.parentNode.insertBefore(match, after);
      subtract += end;
      node = after;
    }
  }
  wordlists_to_positions(word_lists, raw_text) {
    let ret = [];
    for (let words of word_lists) {
      if (words.length === 0) {
        ret.push([]);
        continue;
      }
      let re = new RegExp("\\b(" + words.join("|") + ")\\b", "gm");
      let temp;
      let list = [];
      while ((temp = re.exec(raw_text)) !== null) {
        list.push([temp.index, temp.index + temp[0].length]);
      }
      ret.push(list);
    }
    return ret;
  }
}
export default Explanation;
