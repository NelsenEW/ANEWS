use serde_json::Value;

#[derive(Clone, Debug)]
pub struct GridPatch {
    pub row_min: usize,
    pub col_min: usize,
    pub before: Vec<bool>,
    pub after: Vec<bool>,
    pub width: usize,
    pub height: usize,
}

#[derive(Clone, Debug)]
pub struct EditAction {
    pub action_type: String,
    pub data: Value,
    pub patches: Vec<GridPatch>,
}

pub struct EditHistory {
    pub undo_stack: Vec<EditAction>,
    pub redo_stack: Vec<EditAction>,
    max_size: usize,
}

impl EditHistory {
    pub fn new(max_size: usize) -> Self {
        Self {
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            max_size,
        }
    }

    pub fn push(&mut self, action: EditAction) {
        self.redo_stack.clear();
        self.undo_stack.push(action);
        if self.undo_stack.len() > self.max_size {
            self.undo_stack.remove(0);
        }
    }

    pub fn undo(&mut self) -> Option<EditAction> {
        let action = self.undo_stack.pop()?;
        self.redo_stack.push(action.clone());
        Some(action)
    }

    pub fn redo(&mut self) -> Option<EditAction> {
        let action = self.redo_stack.pop()?;
        self.undo_stack.push(action.clone());
        Some(action)
    }

    pub fn list_actions(&self) -> Vec<Value> {
        self.undo_stack
            .iter()
            .enumerate()
            .map(|(i, a)| serde_json::json!({"type": a.action_type, "index": i}))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_push_and_undo() {
        let mut history = EditHistory::new(50);
        history.push(EditAction {
            action_type: "map_edit".into(),
            data: serde_json::json!({}),
            patches: vec![],
        });
        assert_eq!(history.undo_stack.len(), 1);
        let action = history.undo().unwrap();
        assert_eq!(action.action_type, "map_edit");
        assert_eq!(history.undo_stack.len(), 0);
        assert_eq!(history.redo_stack.len(), 1);
    }

    #[test]
    fn test_redo_after_undo() {
        let mut history = EditHistory::new(50);
        history.push(EditAction {
            action_type: "test".into(),
            data: serde_json::json!({}),
            patches: vec![],
        });
        history.undo();
        let action = history.redo().unwrap();
        assert_eq!(action.action_type, "test");
        assert_eq!(history.undo_stack.len(), 1);
        assert_eq!(history.redo_stack.len(), 0);
    }

    #[test]
    fn test_push_clears_redo() {
        let mut history = EditHistory::new(50);
        history.push(EditAction {
            action_type: "a".into(),
            data: serde_json::json!({}),
            patches: vec![],
        });
        history.undo();
        assert_eq!(history.redo_stack.len(), 1);
        history.push(EditAction {
            action_type: "b".into(),
            data: serde_json::json!({}),
            patches: vec![],
        });
        assert_eq!(history.redo_stack.len(), 0);
    }

    #[test]
    fn test_max_size_evicts_oldest() {
        let mut history = EditHistory::new(3);
        for i in 0..5 {
            history.push(EditAction {
                action_type: format!("action_{i}"),
                data: serde_json::json!({}),
                patches: vec![],
            });
        }
        assert_eq!(history.undo_stack.len(), 3);
        assert_eq!(history.undo_stack[0].action_type, "action_2");
    }
}
