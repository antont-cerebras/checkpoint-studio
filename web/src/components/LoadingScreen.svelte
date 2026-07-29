<script lang="ts">
  // The one screen shown while there is no checkpoint content yet — whichever step is running.
  //
  // Replaces three near-identical panes that each said something different (see
  // `lib/loadstep.ts`). The bar itself is still `LoadingBar`, so a checkpoint wait looks like
  // every other wait in the app; what this adds is *which* step and *whose* work it is.
  import { stepDetail, stepLabel, stepSubject, type LoadStep } from '../lib/loadstep';
  import { proxyHost } from '../stores/server';
  import LoadingBar from './LoadingBar.svelte';

  export let step: LoadStep;

  // The resolved address, not the `:` shorthand: only the server knows which host `:` names, and
  // it has told us (see `resolvedSpec`).
  $: subject = stepSubject(step, $proxyHost);
  $: detail = stepDetail(step, $proxyHost);
</script>

<div class="screen">
  <LoadingBar label={stepLabel(step)} progress={step.progress} />
  <!-- Who is working and why it takes what it takes. The three steps drag for unrelated
       reasons — a slow disk on the server, a slow link to the browser, a large tally — and a
       wait you can attribute is one you can act on. -->
  <p class="detail dim">{detail}</p>
  {#if subject}
    <p class="subject mono" title={subject}>{subject}</p>
  {/if}
</div>

<style>
  .screen {
    padding: 24px;
    display: flex;
    flex-direction: column;
    gap: 6px;
    align-items: flex-start;
  }
  .detail {
    margin: 0;
    font-size: 12px;
  }
  /* The path can be long; keep it on one line and let the tooltip carry the rest, rather than
     letting it set the width of the pane. */
  .subject {
    margin: 2px 0 0;
    font-size: 12px;
    color: var(--fg-dim);
    max-width: min(100%, 80ch);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
</style>
