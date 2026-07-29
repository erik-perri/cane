use std::time::Duration;
use tokio::sync::watch;
use tokio::time::{Instant, sleep_until};

#[derive(Clone, Debug)]
pub struct CommandDeadline {
    deadline: watch::Sender<Instant>,
}

impl CommandDeadline {
    pub fn after(duration: Duration) -> Self {
        let (deadline, _) = watch::channel(Instant::now() + duration);
        Self { deadline }
    }

    pub fn at(&self) -> Instant {
        *self.deadline.borrow()
    }

    pub async fn elapsed(&self) {
        let mut updates = self.deadline.subscribe();

        loop {
            let target = *updates.borrow_and_update();
            tokio::select! {
                _ = sleep_until(target) => {
                    if *updates.borrow() == target {
                        return;
                    }
                }
                changed = updates.changed() => {
                    if changed.is_err() {
                        return;
                    }
                }
            }
        }
    }

    pub fn reset_after(&self, duration: Duration) {
        self.deadline.send_replace(Instant::now() + duration);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test(start_paused = true)]
    async fn reset_after_replaces_the_active_deadline() {
        // Arrange
        let deadline = CommandDeadline::after(Duration::from_secs(10));
        let elapsed_deadline = deadline.clone();
        let elapsed = tokio::spawn(async move {
            elapsed_deadline.elapsed().await;
        });

        // Act
        tokio::time::advance(Duration::from_secs(9)).await;
        tokio::task::yield_now().await;
        deadline.reset_after(Duration::from_secs(10));
        tokio::time::advance(Duration::from_secs(9)).await;
        tokio::task::yield_now().await;

        // Assert
        assert!(!elapsed.is_finished());

        // Act
        tokio::time::advance(Duration::from_secs(1)).await;
        tokio::task::yield_now().await;

        // Assert
        assert!(elapsed.is_finished());
        elapsed.await.unwrap();
    }
}
