use quote::{format_ident, quote, ToTokens};
use std::collections::{HashMap, VecDeque};
use syn::spanned::Spanned;

#[derive(Debug, Default)]
struct State {
    forward_stmts: VecDeque<proc_macro2::TokenStream>,
    backward_stmts: VecDeque<proc_macro2::TokenStream>,
    tracking_args: HashMap<String, (proc_macro2::Span, bool)>,
    ident_ind: u64,
    convert_loop: bool,
}

impl State {
    fn get_new_ident(&mut self) -> syn::Ident {
        let ind = self.ident_ind;
        self.ident_ind += 1;
        format_ident!("__{}", ind)
    }

    fn convert_ad_args_pat_main(
        &mut self,
        pat: &syn::Pat,
        symbols: &mut Vec<(String, proc_macro2::Span, bool)>,
    ) -> syn::Result<(proc_macro2::TokenStream, proc_macro2::TokenStream)> {
        match pat {
            syn::Pat::Type(syn::PatType { attrs, pat, .. }) if attrs.is_empty() => {
                self.convert_ad_args_pat_main(pat, symbols)
            }
            syn::Pat::Tuple(syn::PatTuple { attrs, elems, .. }) if attrs.is_empty() => {
                let (ids_w_mut, ids_wo_mut): (Vec<_>, Vec<_>) = elems
                    .iter()
                    .map(|pat| self.convert_ad_args_pat_main(pat, symbols))
                    .collect::<Result<Vec<_>, _>>()?
                    .into_iter()
                    .unzip();
                Ok((quote! {(#(#ids_w_mut,)*)}, quote! {(#(#ids_wo_mut,)*)}))
            }
            syn::Pat::Ident(syn::PatIdent {
                attrs,
                by_ref: None,
                mutability,
                subpat: None,
                ident,
            }) if attrs.is_empty() => {
                symbols.push((ident.to_string(), ident.span(), mutability.is_some()));
                Ok((quote! {#mutability #ident}, quote! {#ident}))
            }
            _ => {
                return Err(syn::Error::new(
                    pat.span(),
                    "the first arg should be tuple of tuple or ident without attribute",
                ))
            }
        }
    }

    fn convert_ad_args_pat(
        &mut self,
        pat: &syn::Pat,
    ) -> syn::Result<(
        proc_macro2::TokenStream,
        proc_macro2::TokenStream,
        Vec<(String, proc_macro2::Span, bool)>,
    )> {
        let mut symbols = Vec::new();
        let (ts_w, ts_wo) = self.convert_ad_args_pat_main(pat, &mut symbols)?;
        Ok((ts_w, ts_wo, symbols))
    }

    fn convert_ad_args_expr_main(
        expr: &syn::Expr,
        symbols: &mut Vec<(String, proc_macro2::Span)>,
    ) -> syn::Result<(proc_macro2::TokenStream, proc_macro2::TokenStream)> {
        match expr {
            syn::Expr::Tuple(syn::ExprTuple { attrs, elems, .. }) if attrs.is_empty() => {
                let (ids_w_mut, ids_wo_mut): (Vec<_>, Vec<_>) = elems
                    .iter()
                    .map(|expr| Self::convert_ad_args_expr_main(expr, symbols))
                    .collect::<Result<Vec<_>, _>>()?
                    .into_iter()
                    .unzip();
                Ok((quote! {(#(#ids_w_mut,)*)}, quote! {(#(#ids_wo_mut,)*)}))
            }
            syn::Expr::Path(syn::ExprPath {
                attrs,
                qself: None,
                path,
            }) if attrs.is_empty() => {
                let ident = path.get_ident().ok_or(syn::Error::new(
                    path.span(),
                    "a path containing colons cannot be converted to an ident",
                ))?;
                symbols.push((ident.to_string(), ident.span()));
                Ok((quote! {mut #ident}, quote! {#ident}))
            }
            _ => {
                return Err(syn::Error::new(
                    expr.span(),
                    "the first arg should be tuple of tuple, ident, or wildcard without attribute",
                ))
            }
        }
    }

    fn convert_ad_args_expr(
        expr: &syn::Expr,
    ) -> syn::Result<(
        proc_macro2::TokenStream,
        proc_macro2::TokenStream,
        Vec<(String, proc_macro2::Span)>,
    )> {
        let mut symbols = Vec::new();
        let (ts_w, ts_wo) = Self::convert_ad_args_expr_main(expr, &mut symbols)?;
        for (ident, span) in &symbols {
            if symbols.iter().filter(|(i, _)| *i == *ident).count() != 1 {
                return Err(syn::Error::new(
                    *span,
                    format!("{} appears more than once", ident),
                ));
            }
        }
        Ok((ts_w, ts_wo, symbols))
    }

    fn is_loop_prelude(stmt: &syn::Stmt) -> bool {
        match stmt {
            syn::Stmt::Local(syn::Local {
                pat:
                    syn::Pat::Ident(syn::PatIdent {
                        by_ref: None,
                        mutability: Some(_),
                        ident,
                        subpat: None,
                        attrs: attrs_pat,
                    }),
                init: None,
                attrs: attrs_loc,
                ..
            }) if attrs_loc.is_empty() && attrs_pat.is_empty() => ident.to_string() == "__",
            _ => false,
        }
    }

    fn extract_ad_args<'a>(pat: &'a syn::Pat) -> Option<&'a syn::Pat> {
        match pat {
            syn::Pat::Tuple(syn::PatTuple { elems, attrs, .. }) if attrs.is_empty() => {
                match (elems.get(0), elems.get(1), elems.get(2)) {
                    (
                        Some(elem),
                        Some(syn::Pat::Ident(syn::PatIdent {
                            by_ref: None,
                            mutability: None,
                            subpat: None,
                            ident,
                            attrs,
                        })),
                        None,
                    ) if attrs.is_empty() => (ident.to_string() == "__").then_some(elem),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    fn convert_stmt(&mut self, stmt: &syn::Stmt) -> syn::Result<()> {
        let err = |span, message| Err(syn::Error::new(span, message));
        let mut skip = || self.forward_stmts.push_back(stmt.to_token_stream());

        let syn::Stmt::Local(syn::Local {
            pat,
            init: Some(init),
            attrs,
            ..
        }) = stmt
        else {
            skip();
            return Ok(());
        };
        if !attrs.is_empty() {
            skip();
            return Ok(());
        }
        let Some(ad_args_out) = Self::extract_ad_args(pat) else {
            skip();
            return Ok(());
        };

        let ad_args_pb_out = {
            if !init.diverge.is_none() {
                return err(
                    init.expr.span(),
                    "diverging init statements are not supported",
                );
            }
            let syn::Expr::Call(ref expr_call) = *init.expr else {
                return err(
                    init.expr.span(),
                    "the right-hand side should be a call statement",
                );
            };
            let Some(arg_first) = expr_call.args.first() else {
                return err(
                    expr_call.span(),
                    "the call statement needs at least one argument",
                );
            };
            let (args, _, symbols) = Self::convert_ad_args_expr(arg_first)?;
            for (ident, span) in symbols {
                if self.tracking_args.remove(&ident).is_none() {
                    return err(span, &format!("The variable '{}' is not tracked", ident));
                }
            }
            args
        };

        let (ad_args_out, ad_args_pb_in) = {
            let (args, args_pb, symbols) = self.convert_ad_args_pat(ad_args_out)?;
            for (ident, span, mutability) in symbols {
                if !self
                    .tracking_args
                    .insert(ident.clone(), (span, mutability))
                    .is_none()
                {
                    return err(span, &format!("The variable '{}' is not consumed", ident));
                }
            }
            (args, args_pb)
        };

        let pb = self.get_new_ident();
        let orig_rhs = &init.expr;
        self.forward_stmts.push_back(quote! {
            let (#ad_args_out, #pb) = #orig_rhs;
        });
        self.backward_stmts.push_front(quote! {
            let #ad_args_pb_out = #pb (#ad_args_pb_in);
        });

        Ok(())
    }

    fn convert_assign_in_loop(
        &self,
        stmt: &syn::Stmt,
        pb_stack: &syn::Ident,
    ) -> syn::Result<(proc_macro2::TokenStream, Option<proc_macro2::TokenStream>)> {
        fn cmp(x: &syn::Expr, y: &syn::Expr) -> bool {
            match (x, y) {
                (
                    syn::Expr::Tuple(syn::ExprTuple { elems: x, .. }),
                    syn::Expr::Tuple(syn::ExprTuple { elems: y, .. }),
                ) => {
                    if !(x.trailing_punct() == y.trailing_punct() && x.len() == y.len()) {
                        return false;
                    }
                    x.iter().zip(y.iter()).all(|(x, y)| cmp(x, y))
                }
                (
                    syn::Expr::Path(syn::ExprPath { path: x, .. }),
                    syn::Expr::Path(syn::ExprPath { path: y, .. }),
                ) => {
                    let Some(y) = y.get_ident() else { return false };
                    x.is_ident(y)
                }
                _ => false,
            }
        }

        'label: {
            match stmt {
                syn::Stmt::Expr(
                    syn::Expr::Assign(syn::ExprAssign {
                        attrs, left, right, ..
                    }),
                    Some(_),
                ) if attrs.is_empty() => match (&**left, &**right) {
                    (
                        syn::Expr::Tuple(syn::ExprTuple {
                            attrs: attrs_left,
                            elems,
                            ..
                        }),
                        syn::Expr::Call(syn::ExprCall {
                            args: rhs_args,
                            attrs: attrs_right,
                            func,
                            ..
                        }),
                    ) if attrs_left.is_empty() && attrs_right.is_empty() => {
                        match (elems.get(0), elems.get(1), elems.get(2), rhs_args.first()) {
                            (
                                Some(lhs_ad_args),
                                Some(syn::Expr::Path(syn::ExprPath {
                                    attrs,
                                    qself: None,
                                    path,
                                })),
                                None,
                                Some(rhs_ad_args),
                            ) if attrs.is_empty() => match path.get_ident() {
                                Some(ident) if ident.to_string() == "__" => {
                                    if !cmp(lhs_ad_args, rhs_ad_args) {
                                        return Err(syn::Error::new(
                                            lhs_ad_args.span(),
                                            "ad arguments on the lhs and rhs must be equal",
                                        ));
                                    }
                                    let (_, _, ids) = Self::convert_ad_args_expr(lhs_ad_args)?;
                                    for (ident, span) in ids {
                                        let Some((_, true)) =
                                            self.tracking_args.get(&ident.to_string())
                                        else {
                                            return Err(syn::Error::new(
                                                span,
                                                format!(
                                                    "{} is either untracked or immutable",
                                                    ident
                                                ),
                                            ));
                                        };
                                    }
                                    return Ok((
                                        quote! {
                                            #pb_stack.push(
                                                {
                                                    let __;
                                                    (#lhs_ad_args, __) = #func (#rhs_args);
                                                    __
                                                }
                                            );
                                        },
                                        Some(quote! {
                                            while let Some(#pb_stack) = #pb_stack.pop() {
                                                #lhs_ad_args = #pb_stack (#lhs_ad_args);
                                            }
                                        }),
                                    ));
                                }
                                _ => break 'label,
                            },
                            _ => break 'label,
                        }
                    }
                    _ => break 'label,
                },
                _ => break 'label,
            }
        }
        Ok((stmt.to_token_stream(), None))
    }

    fn convert_loop_body(&mut self, stmt: &syn::Stmt) -> syn::Result<()> {
        let pb_stack = self.get_new_ident();

        let convert_body = |body: &syn::Block| {
            let (forward_stmts, backward_stmts): (Vec<_>, Vec<_>) = body
                .stmts
                .iter()
                .map(|s| self.convert_assign_in_loop(s, &pb_stack))
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .unzip();
            let mut backward_stmts: Vec<_> = backward_stmts
                .into_iter()
                .filter_map(std::convert::identity)
                .collect();
            match (backward_stmts.pop(), backward_stmts.pop()) {
                (Some(backward_stmt), None) => Ok((forward_stmts, backward_stmt)),
                _ => Err(syn::Error::new(
                    body.span(),
                    "the loop block must contain exactly one ad statement",
                )),
            }
        };

        match stmt {
            syn::Stmt::Expr(
                syn::Expr::Loop(syn::ExprLoop {
                    attrs, label, body, ..
                }),
                _,
            ) if attrs.is_empty() => {
                let (forward_stmts, backward_stmt) = convert_body(body)?;
                self.forward_stmts.push_back(quote! {
                    let mut #pb_stack = Vec::new();
                    #label loop {
                        #(#forward_stmts)*
                    }
                });
                self.backward_stmts.push_front(backward_stmt);
                Ok(())
            }
            syn::Stmt::Expr(
                syn::Expr::ForLoop(syn::ExprForLoop {
                    attrs,
                    label,
                    pat,
                    expr,
                    body,
                    ..
                }),
                _,
            ) if attrs.is_empty() => {
                let (forward_stmts, backward_stmt) = convert_body(body)?;
                self.forward_stmts.push_back(quote! {
                    let mut #pb_stack = Vec::new();
                    #label for #pat in #expr {
                        #(#forward_stmts)*
                    }
                });
                self.backward_stmts.push_front(backward_stmt);
                Ok(())
            }
            syn::Stmt::Expr(
                syn::Expr::While(syn::ExprWhile {
                    attrs,
                    label,
                    cond,
                    body,
                    ..
                }),
                _,
            ) if attrs.is_empty() => {
                let (forward_stmts, backward_stmt) = convert_body(body)?;
                self.forward_stmts.push_back(quote! {
                    let mut #pb_stack = Vec::new();
                    #label while #cond {
                        #(#forward_stmts)*
                    }
                });
                self.backward_stmts.push_front(backward_stmt);
                Ok(())
            }
            _ => {
                return Err(syn::Error::new(
                    stmt.span(),
                    "the next statement after 'let mut __;' must be a loop block",
                ))
            }
        }
    }

    fn convert(&mut self, expr: &syn::ExprClosure) -> syn::Result<proc_macro2::TokenStream> {
        let err = |span, message| Err(syn::Error::new(span, message));

        let syn::ExprClosure {
            attrs,
            lifetimes: None,
            constness: None,
            movability: None,
            asyncness: None,
            capture,
            inputs,
            output: syn::ReturnType::Default,
            body,
            ..
        } = expr
        else {
            return err(
                expr.span(),
                "static closures, async closures, const closures, \
                closures with lifetimes, and closures with return types are not supported",
            );
        };
        for attr in attrs {
            return err(attr.span(), "the attribute of the closure is not supported");
        }

        // input
        let (input_first, input_ty, backward_output, input_rest) = {
            let mut iter = inputs.iter();
            let Some(pat) = iter.next() else {
                return err(inputs.span(), "the input of the closure cannot be empty");
            };
            let (input_first, backward_output, symbols) = self.convert_ad_args_pat(pat)?;
            let ty = match pat {
                syn::Pat::Type(syn::PatType { ty, .. }) => Some(quote! {:#ty}),
                _ => None,
            };
            for (ident, span, mutability) in symbols {
                self.tracking_args.insert(ident, (span, mutability));
            }
            (input_first, ty, backward_output, iter)
        };

        // body
        let (output, backward_input) = {
            let syn::Expr::Block(syn::ExprBlock {
                block: syn::Block { stmts, .. },
                ..
            }) = &**body
            else {
                return err(body.span(), "The closure must include a block");
            };
            let Some((syn::Stmt::Expr(last_stmt, None), stmts)) = stmts.split_last() else {
                return err(body.span(), "The block must contain at least one line");
            };

            for stmt in stmts {
                if self.convert_loop {
                    self.convert_loop = false;
                    self.convert_loop_body(stmt)?;
                } else {
                    if Self::is_loop_prelude(stmt) {
                        self.convert_loop = true;
                    } else {
                        self.convert_stmt(stmt)?
                    }
                }
            }
            let (backward_input, output, return_symbols) = Self::convert_ad_args_expr(&last_stmt)?;

            // check
            for (ident, (span, _)) in self.tracking_args.iter() {
                if return_symbols.iter().find(|(i, _)| i == ident).is_none() {
                    return err(*span, &format!("The variable '{}' is not consumed", ident));
                };
            }
            for (ident, span) in return_symbols.iter() {
                if !self.tracking_args.contains_key(ident) {
                    return err(*span, &format!("The variable '{}' is not tracked", ident));
                };
            }

            (output, backward_input)
        };

        let forward_stmts = &self.forward_stmts.make_contiguous();
        let backward_stmts = &self.backward_stmts.make_contiguous();

        Ok(quote! {
            #capture | #input_first #input_ty #(, #input_rest)* | {
                #(#forward_stmts)*
                (#output, move | #backward_input | {
                    #(#backward_stmts)*
                    #backward_output
                })
            }
        })
    }
}

#[proc_macro]
pub fn ad(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    State::default()
        .convert(&syn::parse_macro_input!(input as syn::ExprClosure))
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}
