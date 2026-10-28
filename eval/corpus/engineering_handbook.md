# Engineering Handbook

## Code Review
Every pull request needs at least 2 approvals, and one of them must come from a code owner. Pull requests should stay under 400 changed lines so they can be reviewed properly. Reviewers aim to respond within 1 business day. All pull requests are merged with squash merge, and the CI pipeline must pass before merging.

## Deployment
Production deployments run on Tuesdays and Thursdays at 14:00 UTC. A holiday deployment freeze applies from December 20 to January 3. Hotfixes outside the schedule need approval from the on-call lead. If the error rate exceeds 2 percent after a deployment, the release must be rolled back within 10 minutes.

## On-Call Rotation
On-call is a weekly rotation that starts on Monday at 09:00, with a primary and a secondary engineer. Pages must be acknowledged within 10 minutes. The on-call allowance is $250 per week. An engineer paged between 22:00 and 06:00 receives a compensatory day off. A handover meeting takes place every Monday morning.
