import d3 from "d3";
class Barchart {
  // svg: d3 object with the svg in question
  // exp_array: list of (feature_name, weight)
  constructor(
    svg,
    exp_array,
    two_sided = true,
    titles = undefined,
    colors = ["red", "green"],
    show_numbers = false,
    bar_height = 5
  ) {
    let svg_width = Math.min(600, parseInt(svg.style("width")));
    let bar_width = two_sided ? svg_width / 2 : svg_width;
    if (titles === undefined) {
      titles = two_sided ? ["Cons", "Pros"] : "Pros";
    }
    if (show_numbers) {
      bar_width = bar_width - 30;
    }
    let x_offset = two_sided ? svg_width / 2 : 10;
    // 13.1 is +- the width of W, the widest letter.
    if (two_sided && titles.length == 2) {
      svg
        .append("text")
        .attr("x", svg_width / 4)
        .attr("y", 15)
        .attr("font-size", "20")
        .attr("text-anchor", "middle")
        .style("fill", colors[0])
        .text(titles[0]);

      svg
        .append("text")
        .attr("x", (svg_width / 4) * 3)
        .attr("y", 15)
        .attr("font-size", "20")
        .attr("text-anchor", "middle")
        .style("fill", colors[1])
        .text(titles[1]);
    } else {
      let pos = two_sided ? svg_width / 2 : x_offset;
      let anchor = two_sided ? "middle" : "begin";
      svg
        .append("text")
        .attr("x", pos)
        .attr("y", 15)
        .attr("font-size", "20")
        .attr("text-anchor", anchor)
        .text(titles);
    }
    let yshift = 20;
    let space_between_bars = 0;
    let text_height = 16;
    let space_between_bar_and_text = 3;
    let total_bar_height =
      text_height +
      space_between_bar_and_text +
      bar_height +
      space_between_bars;
    let total_height = total_bar_height * exp_array.length;
    this.svg_height = total_height + yshift;
    let yscale = d3.scale
      .linear()
      .domain([0, exp_array.length])
      .range([yshift, yshift + total_height]);
    let names = exp_array.map(v => v[0]);
    let weights = exp_array.map(v => v[1]);
    let max_weight = Math.max(...weights.map(v => Math.abs(v)));
    let xscale = d3.scale
      .linear()
      .domain([0, Math.max(1, max_weight)])
      .range([0, bar_width]);

    for (var i = 0; i < exp_array.length; ++i) {
      let name = names[i];
      let weight = weights[i];
      var size = xscale(Math.abs(weight));
      let to_the_right = weight > 0 || !two_sided;
      let text = svg
        .append("text")
        .attr("x", to_the_right ? x_offset + 2 : x_offset - 2)
        .attr("y", yscale(i) + text_height)
        .attr("text-anchor", to_the_right ? "begin" : "end")
        .attr("font-size", "14")
        .text(name);
      while (text.node().getBBox()["width"] + 1 > bar_width) {
        let cur_text = text.text().slice(0, text.text().length - 5);
        text.text(cur_text + "...");
        if (text === "...") {
          break;
        }
      }
      let bar = svg
        .append("rect")
        .attr("height", bar_height)
        .attr("x", to_the_right ? x_offset : x_offset - size)
        .attr("y", text_height + yscale(i) + space_between_bar_and_text) // + bar_height)
        .attr("width", size)
        .style("fill", weight > 0 ? colors[1] : colors[0]);
      if (show_numbers) {
        let bartext = svg
          .append("text")
          .attr("x", to_the_right ? x_offset + size + 1 : x_offset - size - 1)
          .attr("text-anchor", weight > 0 || !two_sided ? "begin" : "end")
          .attr(
            "y",
            bar_height + yscale(i) + text_height + space_between_bar_and_text
          )
          .attr("font-size", "10")
          .text(Math.abs(weight).toFixed(2));
      }
    }
    let line = svg
      .append("line")
      .attr("x1", x_offset)
      .attr("x2", x_offset)
      .attr("y1", bar_height + yshift)
      .attr("y2", Math.max(bar_height, yscale(exp_array.length)))
      .style("stroke-width", 2)
      .style("stroke", "black");
  }
}
export default Barchart;
