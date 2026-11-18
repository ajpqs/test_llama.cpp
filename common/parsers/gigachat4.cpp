#include "parsers.h"

// the template pairs tool results with tool calls by position, so put them in tool call order (matched by id)
static json gigachat4_sort_tool_results(const json & messages) {
    json adjusted = messages;

    for (size_t i = 0; i < adjusted.size(); i++) {
        const auto & msg = adjusted[i];
        if (msg.value("role", "") != "assistant" || !msg.contains("tool_calls") || !msg.at("tool_calls").is_array()) {
            continue;
        }

        std::map<std::string, size_t> call_order;
        const auto & tool_calls = msg.at("tool_calls");
        for (size_t idx = 0; idx < tool_calls.size(); idx++) {
            call_order[tool_calls[idx].value("id", "")] = idx;
        }

        // a run with missing, unknown or repeated ids is left as it is
        bool sortable = call_order.size() == tool_calls.size() && call_order.count("") == 0;
        std::vector<json> results;
        std::vector<bool> answered(tool_calls.size(), false);
        for (size_t k = i + 1; k < adjusted.size() && adjusted[k].value("role", "") == "tool"; k++) {
            const auto it = call_order.find(adjusted[k].value("tool_call_id", ""));
            sortable = sortable && it != call_order.end() && !answered[it->second];
            if (sortable) {
                answered[it->second] = true;
            }
            results.push_back(adjusted[k]);
        }

        if (sortable && results.size() > 1) {
            std::stable_sort(results.begin(), results.end(), [&](const json & a, const json & b) {
                return call_order.at(a.value("tool_call_id", "")) < call_order.at(b.value("tool_call_id", ""));
            });
            for (size_t k = 0; k < results.size(); k++) {
                adjusted[i + 1 + k] = std::move(results[k]);
            }
        }
        i += results.size();
    }

    return adjusted;
}

common_chat_params common_chat_params_init_gigachat4(
        const common_chat_template & tmpl,
        const autoparser::generation_params & inputs) {

    auto render_inputs = inputs;
    render_inputs.messages = gigachat4_sort_tool_results(inputs.messages);
    if (!render_inputs.extra_context.is_object()) {
        render_inputs.extra_context = json::object();
    }
    if (!render_inputs.extra_context.contains("reasoning")) {
        render_inputs.extra_context["reasoning"] = inputs.enable_thinking;
    }

    const std::string ROLE_SEP     = "<|role_sep|>\n";
    const std::string GEN_PROMPT   = "assistant" + ROLE_SEP;
    const std::string GCML         = "｜GCML｜";
    const std::string THINK_START  = "<think>";
    const std::string THINK_END    = "</think>";
    const std::string FC_START     = "<" + GCML + "tool_calls>";
    const std::string FC_END       = "</" + GCML + "tool_calls>";
    const std::string INVOKE_START = "<" + GCML + "invoke";
    const std::string INVOKE_END   = "</" + GCML + "invoke>";
    const std::string PARAM_START  = "<" + GCML + "parameter";
    const std::string PARAM_END    = "</" + GCML + "parameter>";

    common_chat_params data;

    data.prompt             = common_chat_template_direct_apply_impl(tmpl, render_inputs);
    data.generation_prompt  = common_chat_template_generation_prompt_impl(tmpl, render_inputs);
    data.format             = COMMON_CHAT_FORMAT_PEG_NATIVE;
    data.supports_thinking  = true;
    data.thinking_start_tag = THINK_START;
    data.thinking_end_tags  = { THINK_END, FC_START };
    data.preserved_tokens   = {
        "<|message_sep|>\n\n",
        ROLE_SEP,
        GCML,
        THINK_START,
        THINK_END,
    };
    data.message_delimiters = {
        { COMMON_CHAT_ROLE_ASSISTANT, "assistant" + ROLE_SEP       },
        { COMMON_CHAT_ROLE_USER,      "user" + ROLE_SEP            },
        { COMMON_CHAT_ROLE_TOOL,      "function result" + ROLE_SEP },
        { COMMON_CHAT_ROLE_SYSTEM,    "system" + ROLE_SEP          },
    };
    // the turn separator is an ordinary token, it does not stop generation
    data.additional_stops   = { "<|message_sep|>" };

    // the reasoning argument as the template tests it, the template turns none into true
    const auto & arg = render_inputs.extra_context.at("reasoning");
    const bool reasoning_on = arg.is_boolean() ? arg.get<bool>()
                            : arg.is_number()  ? arg.get<double>() != 0.0
                            : arg.is_string()  ? !arg.get<std::string>().empty()
                            : arg.is_null() || !arg.empty();

    // a continuation of the reasoning reopens the think block whatever the reasoning argument
    const bool think_open = reasoning_on ||
        (render_inputs.has_continuation() && render_inputs.continue_final_message == COMMON_CHAT_CONTINUATION_REASONING);

    if (render_inputs.has_continuation()) {
        const auto & msg = render_inputs.continue_msg;

        data.generation_prompt = GEN_PROMPT;
        if (render_inputs.continue_final_message == COMMON_CHAT_CONTINUATION_REASONING) {
            data.generation_prompt += THINK_START + msg.reasoning_content;
        } else {
            // the template writes a think block in every assistant message, with its text only when reasoning is on
            data.generation_prompt += THINK_START + (reasoning_on ? msg.reasoning_content : "") + THINK_END + "\n\n";
            data.generation_prompt += msg.render_content();
        }

        data.prompt.append(data.generation_prompt);
    }

    auto has_tools           = render_inputs.tools.is_array() && !render_inputs.tools.empty();
    auto has_response_format = render_inputs.json_schema.is_object();
    auto extract_reasoning   = render_inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;
    auto include_grammar     = has_response_format || (has_tools && render_inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE);

    data.parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        // with reasoning off the template closes an empty think block right after the assistant header, it is part of the prompt
        auto header = p.literal(GEN_PROMPT);
        if (!think_open) {
            header = p.literal(GEN_PROMPT + THINK_START + THINK_END + "\n\n") | header;
        }
        // the template writes no assistant header when only the system message is given
        auto generation_prompt = p.optional(header);
        auto end               = p.end();

        auto think_block = [&](const common_peg_parser & body) {
            return p.optional(THINK_START + body + THINK_END + p.optional(p.literal("\n\n")));
        };
        auto reasoning = extract_reasoning && think_open ? think_block(p.reasoning(p.until(THINK_END))) : p.eps();

        if (has_response_format) {
            // reasoning_format=none keeps the think block in the content
            auto thoughts = extract_reasoning ? reasoning : p.content(think_block(p.until(THINK_END)));
            return generation_prompt + thoughts + p.content(p.schema(p.json(), "response-format", render_inputs.json_schema)) + end;
        }

        if (!has_tools || render_inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_NONE) {
            return generation_prompt + reasoning + p.content(p.rest()) + end;
        }

        auto tool_choice = p.choice();
        foreach_function(render_inputs.tools, [&](size_t tool_index, const json & tool) {
            const auto & function = tool.at("function");
            std::string  name     = function.at("name");
            const std::string tool_rule = "tool-" + std::to_string(tool_index);

            std::vector<common_peg_parser> required_parsers;
            std::vector<common_peg_parser> optional_parsers;
            foreach_parameter(function, [&](size_t param_index, const common_chat_schema_property & param, const common_chat_schema_document_ptr & doc) {
                // string="true" carries a raw string, string="false" a JSON value
                const auto types = param.schema->value_types();
                const std::string arg_rule = tool_rule + "-arg-" + std::to_string(param_index);

                auto arg_open = p.literal(PARAM_START + " name=\"") +
                    p.tool_arg_name(p.literal(param.name)) + p.literal("\" string=\"");
                auto arg_value = p.choice();
                if (types.has(common_chat_schema::TYPE_STRING)) {
                    arg_value |= p.tool_arg_open(arg_open + p.literal("true\">")) +
                        p.tool_arg_string_value(p.until(PARAM_END));
                }
                if (!types.is_only(common_chat_schema::TYPE_STRING)) {
                    arg_value |= p.tool_arg_open(arg_open + p.literal("false\">")) +
                        p.tool_arg_json_value(p.schema(p.json(), arg_rule + "-schema", doc, *param.schema));
                }
                auto arg = p.tool_arg(arg_value + p.tool_arg_close(p.literal(PARAM_END)));

                auto named_arg = p.rule(arg_rule, arg);
                if (param.required) {
                    required_parsers.push_back(named_arg);
                } else {
                    optional_parsers.push_back(named_arg);
                }
            });

            auto optional_args = p.eps();
            if (!optional_parsers.empty()) {
                optional_args = p.zero_or_more(p.choice(optional_parsers) + p.space());
            }
            for (auto & arg : required_parsers) {
                arg = arg + p.space() + optional_args;
            }
            auto args_seq = p.space() + optional_args + p.permute(tool_rule + "-args", required_parsers);

            auto func_parser = p.tool(
                p.tool_open(p.literal(INVOKE_START + " name=\"") +
                            p.tool_name(p.literal(name)) + p.literal("\">\n")) +
                args_seq + p.space() +
                p.tool_close(p.literal(INVOKE_END)));

            tool_choice |= p.rule(tool_rule, func_parser);
        });

        auto require_tools = render_inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED;

        common_peg_parser tool_calls = p.eps();
        if (render_inputs.parallel_tool_calls) {
            tool_calls = p.trigger_rule("tool-call",
                p.literal(FC_START) + p.space() + tool_choice +
                p.zero_or_more(p.space() + tool_choice) + p.space() + p.literal(FC_END));
        } else {
            tool_calls = p.trigger_rule("tool-call",
                p.literal(FC_START) + p.space() + tool_choice + p.space() + p.literal(FC_END));
        }

        auto reasoning_with_tc = THINK_START +
            p.reasoning(p.until_one_of({ "\n\n" + FC_START, FC_START, THINK_END })) +
            p.optional(p.literal("\n\n")) + tool_calls;

        if (!require_tools) {
            tool_calls = p.optional(tool_calls);
        }

        // the template separates the content from the tool calls with two newlines, stop the content before them
        auto content_before_tools = p.content(p.until_one_of({ "\n\n" + FC_START, FC_START })) + p.optional(p.literal("\n\n"));
        if (extract_reasoning && think_open) {
            return generation_prompt + (reasoning_with_tc | (reasoning + content_before_tools + tool_calls)) + end;
        }
        return generation_prompt + reasoning + content_before_tools + tool_calls + end;
    });

    if (include_grammar) {
        data.grammar_lazy = !has_response_format && render_inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_AUTO;
        data.grammar      = build_grammar([&](const common_grammar_builder & builder) {
            data.parser.build_grammar(builder, data.grammar_lazy);
        });

        if (!has_response_format) {
            data.grammar_triggers = {
                { COMMON_GRAMMAR_TRIGGER_TYPE_WORD, FC_START },
            };
        }
    }

    return data;
}
